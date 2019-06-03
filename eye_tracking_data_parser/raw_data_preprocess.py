
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
        subjectIntId = tempList[0]
        print('Log.....Getting Ascii file data - ' + ascFileName)
        ascFile = codecs.open(asc_directory + '//' + ascFileName, encoding='utf-8-sig')
        ascData = ascFile.readlines()
        # Split data to columns and get only relevnt ones
        ascDf = pd.DataFrame(ascData, columns=['data'])
        ascDf = pd.DataFrame([x.split('\t') for x in list(ascDf['data'])])
        ascDf = ascDf[[0, 1, 2, 4, 5]]
        print('Log.....Getting txt file data for subject id - ' + subjectIntId)
        # Read txt file of subject same subject as asc file
        txtFileName = os.path.basename(glob.glob(txt_directory + '//*' + subjectIntId +'_Scale'+ '*txt')[0])
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
            trialData['trial'] = trial + 1
            mergeData = pd.merge(txtData, trialData, on='trial')
            if (trial + 1 == 1):
                allTrialsData = pd.DataFrame(columns=mergeData.columns)
            allTrialsData = pd.concat([allTrialsData, mergeData])
            print('Log.....' + 'Trial' + str(trial + 1).zfill(3))

        # get only dominant eye data
        txtFilePersonalDataName = os.path.basename(
            glob.glob(txt_directory + '//*' + subjectIntId + '_personalDetails' + '*txt')[0])
        txtpersonalData = pd.read_table(txt_directory + '//' + txtFilePersonalDataName)
        dominant_eye = txtpersonalData['dominant eye (1-right, 2-left)'].values[0]
        if dominant_eye == 1:
            allTrialsData = allTrialsData.drop([1, 2], axis=1)
            allTrialsData.rename(columns={4: 1, 5: 2}, inplace=True)
        else:
            allTrialsData = allTrialsData.drop([4, 5], axis=1)

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
    current_path = os.getcwd() # update if needed
    data_csv_path = current_path + csv_file_name
    #returns path for data csv file
    return data_csv_path


def find_stim_boundaries(screen_resolution, stim_resolution):
    min_x = (screen_resolution[1]/2) - (stim_resolution[1]/2)
    max_x = (screen_resolution[1]/2) + (stim_resolution[1]/2)
    min_y = (screen_resolution[0]/2) - (stim_resolution[0]/2)
    max_y = (screen_resolution[0]/2) + (stim_resolution[0]/2)

    return min_x, max_x, min_y, max_y


def data_tidying(df, screen_resolution):
    print('Log..... Data tidying')
    # Change X, Y and timeStamp data from String to Numeric changing strings " . ", "EBLINK", FIX", "SACC" to nan
    df.X_axis = pd.to_numeric(df.X_axis, errors='coerce')
    df.Y_axis = pd.to_numeric(df.Y_axis, errors='coerce')
    df.timeStamp = pd.to_numeric(df.timeStamp, errors='coerce')
    df.X_axis = df.X_axis.round()
    df.Y_axis = df.Y_axis.round()
    df.bid = df.bid.round()
    # Remove Nan
    df.dropna(inplace=True)
    df.X_axis = df.X_axis.astype(int)
    df.Y_axis = df.Y_axis.astype(int)
    df = df[df.bid != 999]
    df.reset_index(drop=True, inplace=True)


    #update subjectID to be unique between expirements (fix should be at the raw_data_to_csv def)
    #df.loc[df.stimId == 1, 'subjectID'] = df['subjectID'].astype(str) + '01'
    # add 'sampleId' field for each uniqe sample
    df['sampleId'] = df['subjectID'].astype(str) + '_' + df['stimName'].astype(str)

    # stim is snack
    stim_id = 2
    stim_resolution = ([520,690])
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    # get only datapoints within stim boundaries
    stimARegionDataDf = df[((df['stimId'] == stim_id) & (df['X_axis'] >= min_x) & (df['X_axis'] <= max_x) &
                         (df['Y_axis'] >= min_y) & (df['Y_axis'] <= max_y)) == True]
    # Shifting x,y datapoint to start from (0,0) point
    stimARegionDataDf.X_axis = stimARegionDataDf.X_axis - min_x
    stimARegionDataDf.Y_axis = stimARegionDataDf.Y_axis - min_y


    #stim face
    stim_resolution = ([600, 480])
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    #get only datapoints within stim boundaries
    stimBRegionDataDf = df[((df['stimId'] != stim_id) & (df['X_axis'] >= min_x) & (df['X_axis'] <= max_x) & (
            df['Y_axis'] >= min_y) & (df['Y_axis'] <= max_y)) == True]
    # Shifting x,y datapoint to start from (0,0) point
    stimBRegionDataDf.X_axis = stimBRegionDataDf.X_axis - min_x
    stimBRegionDataDf.Y_axis = stimBRegionDataDf.Y_axis - min_y

    byImgRegionDataDf = pd.concat([stimARegionDataDf,stimBRegionDataDf])
    byImgRegionDataDf.reset_index(drop=True, inplace=True)
    byImgRegionDataDf.drop(byImgRegionDataDf.columns[[0]], axis=1, inplace=True)


    return byImgRegionDataDf

