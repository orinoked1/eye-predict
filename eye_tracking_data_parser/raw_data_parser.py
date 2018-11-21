
import pandas as pd
import codecs
import os
import glob

unpickled_df = pd.read_pickle("bdm_bmm_short_data_df")

flag = 0
# Set directory name that contains output directory with asc and txt files
asc_directory = '../row_data/bdm_bmm_short_data/output/asc'
txt_directory = '../row_data/bdm_bmm_short_data/output/txt'
# Set string represents trail start data records
indexStartStr = 'TrialStart'  # 'TrialStart'
# Set string represents trail ends data records
indexEndStr = 'Response'  # 'ScaleStart'


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
    ascDf = ascDf['data'].apply(lambda x: pd.Series(x.split('\t')))
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
        #trialData['subjectID'] = subjectIntId
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
allSubjectsData.columns = ['subjectID','trialNum','onsettime','stimName','bid','RT','stimType','stimId',
                           'timeStamp','X_axis','Y_axis' ]


#Store all subjects data DF as pikle
allSubjectsData.to_pickle('bdm_bmm_short_data_df')

#Read pikle into DF
#file_name = 'bdm_bmm_short_data_df'
#df = pd.read_pickle(file_name)