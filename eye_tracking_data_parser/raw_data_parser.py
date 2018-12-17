
import pandas as pd
import codecs
import os
import glob


def raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name):
    flag = 0
    # Set directory name that contains output directory with asc and txt files
    asc_directory = asc_files_path #'../row_data/scale_ranking_bmm_short_data/output/asc'
    txt_directory = txt_files_path #'../row_data/scale_ranking_bmm_short_data/output/txt'
    # Set string represents trail start data records
    indexStartStr = trial_satart_str #'TrialStart'
    # Set string represents trail ends data records
    indexEndStr = trial_end_str #'ScaleStart'

    # Run over each Ascii file and open it
    for ascFile in glob.glob(asc_directory + '//*asc'):
        ascFileName = os.path.basename(ascFile)
        tempList = ascFileName.split('_')
        subjectIntId = tempList[2]
        print('Log.....Getting Ascii file data - ' + ascFileName)
        ascFile = codecs.open(asc_directory + '//' + ascFileName, encoding='utf-8-sig')
        ascData = ascFile.readlines()
        # Split data to columns and get only relevnt ones
        ascDf = pd.DataFrame(ascData, columns=['data'])
        ascDf = pd.DataFrame([x.split('\t') for x in list(ascDf['data'])])
        # below a very slow option for spliting the data:
        # ascDf = ascDf['data'].apply(lambda x: pd.Series(x.split('\t')))
        ascDf = ascDf[[0, 1, 2]]
        print('Log.....Getting txt file data for subject id - ' + subjectIntId)
        # Read txt file of subject same subject as asc file
        txtFileName = os.path.basename(glob.glob(txt_directory + '//*' + subjectIntId + '*txt')[0])
        txtData = pd.read_table(txt_directory + '//' + txtFileName)
        # Get number of trials and subject ID
        trialCount = txtData.count()[0]
        # subjectId = txtData['subjectID'][0]
        print('Log.....Runing over all trials of subject id - ' + subjectIntId)
        # Run over all trials per user and merge the asc data with the txt data
        for trial in range(trialCount):
            trial_str = 'Trial' + str(trial + 1).zfill(3)
            indexStart = ascDf[ascDf[1].str.contains(indexStartStr, na=False) &
                               ascDf[1].str.contains(trial_str, na=False)].index[0] + 1
            indexEnd = ascDf[ascDf[1].str.contains(indexEndStr, na=False) &
                             ascDf[1].str.contains(trial_str, na=False)].index[0] - 1
            # Get the data, starting from 'TrialStart' to subjects 'Response'
            trialData = ascDf.loc[indexStart:indexEnd]
            # trialData['subjectID'] = subjectIntId
            trialData['runtrial'] = trial + 1
            mergeData = pd.merge(txtData, trialData, on='runtrial')
            if (trial + 1 == 1):
                allTrialsData = pd.DataFrame(columns=mergeData.columns)
            allTrialsData = pd.concat([allTrialsData, mergeData])
            print('Log.....' + 'Trial' + str(trial + 1).zfill(3))

        # Appending all data to one DataFrame
        if flag == 0:
            allSubjectsData = pd.DataFrame(columns=allTrialsData.columns)
            flag = 1
        allSubjectsData = pd.concat([allSubjectsData, allTrialsData])

    # Rename columns if needed
    allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                               'timeStamp', 'X_axis', 'Y_axis']

    # Store all subjects data DF as CSV
    allSubjectsData.to_csv(csv_file_name) #'scale_ranking_bmm_short_data_df.csv'

    data_csv_path = 'eye_tracking_data_parser/' + csv_file_name
    #returns path for data csv file
    return data_csv_path

def data_tidying(df):
    # Remove Nan
    tidyDf = df.dropna(how='any')
    # remove strings " . ", "EBLINK", FIX", "SACC" data rows
    tidyDf = tidyDf[(tidyDf['timeStamp'].str.contains('EFIX', na=False) == False) &
                    (tidyDf['timeStamp'].str.contains('EBLINK', na=False) == False) &
                    (tidyDf['timeStamp'].str.contains('ESACC', na=False) == False) &
                    (tidyDf['timeStamp'].str.contains(' . ', na=False) == False)]
    tidyDf = tidyDf.reset_index()
    # Change X, Y and timeStamp data from String to Numeric
    tidyDf.X_axis = pd.to_numeric(tidyDf.X_axis, errors='coerce')
    tidyDf.Y_axis = pd.to_numeric(tidyDf.Y_axis, errors='coerce')
    tidyDf.timeStamp = pd.to_numeric(tidyDf.timeStamp, errors='coerce')
    # add 'sampleId' field for each uniqe sample
    tidyDf['sampleId'] = tidyDf['subjectID'].astype(str) + '_' + tidyDf['trialNum'].astype(str) + '_' + tidyDf[
        'stimId'].astype(str)
    # Adjust the x,y data points to be within the image frame and in scale of 0-400, rounded
    tidyDf = tidyDf[(tidyDf['X_axis'] > 760) &
                                                                (tidyDf['X_axis'] < 1160) &
                                                                (tidyDf['Y_axis'] > 340) &
                                                                (tidyDf['Y_axis'] < 740)]

    tidyDf.X_axis = tidyDf.X_axis - 760
    tidyDf.Y_axis = tidyDf.Y_axis - 340
    tidyDf.X_axis = tidyDf.X_axis.round()
    tidyDf.Y_axis = tidyDf.Y_axis.round()

    return tidyDf