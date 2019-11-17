"""
Original code for runing my experiment data and process it.
The code is relevant for both eyes recorded.
For the analysis you should use the func - 'data_tidying_for_analysis'
"""
import pandas as pd
import codecs
import os
import glob

EYE_TRACKER_SAMPLE_RATE = 3000


def raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name):
    flag = 0
    # Set directory name that contains output directory with asc and txt files
    asc_directory = asc_files_path
    txt_directory = txt_files_path
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
        ascDf = ascDf[[0, 1, 2, 3, 4, 5, 6]]
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
            #allTrialsData = allTrialsData.drop([1, 2, 3], axis=1)
            #allTrialsData.rename(columns={4: 1, 5: 2, 6: 3}, inplace=True)
            allTrialsData['dominant_eye'] = 'R'
        else:
            #allTrialsData = allTrialsData.drop([4, 5, 6], axis=1)
            allTrialsData['dominant_eye'] = 'L'

        # Appending all data to one DataFrame
        if flag == 0:
            allSubjectsData = pd.DataFrame(columns=allTrialsData.columns)
            flag = 1
        allSubjectsData = pd.concat([allSubjectsData, allTrialsData])

    # Rename columns if needed
    allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                               'timeStamp', 'L_X_axis', 'L_Y_axis', 'L_pupil_size', 'R_X_axis', 'R_Y_axis', 'R_pupil_size', 'dominant_eye']

    # Store all subjects data DF as CSV
    allSubjectsData.to_csv(csv_file_name) #'scale_ranking_bmm_short_data_df.csv'
    current_path = os.getcwd() # update if needed
    data_csv_path = current_path + csv_file_name
    #returns path for data csv file
    return data_csv_path


def find_stim_boundaries(screen_resolution, stim_resolution):
    min_x = (screen_resolution[1]/2) - (stim_resolution[0]/2)
    max_x = (screen_resolution[1]/2) + (stim_resolution[0]/2)
    min_y = (screen_resolution[0]/2) - (stim_resolution[1]/2)
    max_y = (screen_resolution[0]/2) + (stim_resolution[1]/2)

    return min_x, max_x, min_y, max_y


def data_tidying_for_dataset_building(df, screen_resolution):
    print('Log..... Data tidying')

    # add 'sampleId' field for each uniqe sample
    df['sampleId'] = df['subjectID'].astype(str) + '_' + df['stimName'].astype(str)
    if 'L_X_axis' in df:
        df.drop(['L_X_axis', 'L_Y_axis', 'L_pupil_size'], inplace=True, axis=1)
    # Change X, Y and timeStamp data from String to Numeric changing strings " . ", "EBLINK", FIX", "SACC" to nan
    df.R_X_axis = pd.to_numeric(df.R_X_axis, errors='coerce')
    df.R_Y_axis = pd.to_numeric(df.R_Y_axis, errors='coerce')
    df.timeStamp = pd.to_numeric(df.timeStamp, errors='coerce')
    df.R_X_axis = df.R_X_axis.round()
    df.R_Y_axis = df.R_Y_axis.round()
    # Remove Nan
    df.dropna(inplace=True)
    df.R_X_axis = df.R_X_axis.astype(int)
    df.R_Y_axis = df.R_Y_axis.astype(int)
    df.timeStamp = df.timeStamp.astype(int)
    df = df[df.bid != 999]
    df.reset_index(drop=True, inplace=True)


    # stim is snack
    stim_id = 2
    stim_resolution = ([690,520])
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    # get only datapoints within stim boundaries
    stimARegionDataDf = df[((df['stimId'] == stim_id) & (df['R_X_axis'] >= min_x) & (df['R_X_axis'] <= max_x) &
                            (df['R_Y_axis'] >= min_y) & (df['R_Y_axis'] <= max_y)) == True]
    # Shifting x,y datapoint to start from (0,0) point
    stimARegionDataDf.R_X_axis = stimARegionDataDf.R_X_axis - min_x
    stimARegionDataDf.R_Y_axis = stimARegionDataDf.R_Y_axis - min_y


    #stim face
    stim_resolution = ([480, 600])
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    #get only datapoints within stim boundaries
    stimBRegionDataDf = df[((df['stimId'] != stim_id) & (df['R_X_axis'] >= min_x) & (df['R_X_axis'] <= max_x) & (
                                   df['R_Y_axis'] >= min_y) & (df['R_Y_axis'] <= max_y)) == True]

    # Shifting x,y datapoint to start from (0,0) point
    stimBRegionDataDf.R_X_axis = stimBRegionDataDf.R_X_axis - min_x
    stimBRegionDataDf.R_Y_axis = stimBRegionDataDf.R_Y_axis - min_y

    byImgRegionDataDf = pd.concat([stimARegionDataDf, stimBRegionDataDf])
    byImgRegionDataDf.reset_index(drop=True, inplace=True)
    byImgRegionDataDf.drop(byImgRegionDataDf.columns[[0]], axis=1, inplace=True)


    return byImgRegionDataDf

def data_tidying_for_analysis(df, screen_resolution, snack_stim_size, face_stim_size):
    print('Log..... Data tidying')

    # Get relevant raw data for fixations and saccades
    Efix_df = df[df['timeStamp'].str.contains('EFIX', na=False)]
    Efix_df.dropna(inplace=True, axis=1)
    fix_action = Efix_df["timeStamp"].str.split(expand=True)
    fix_df = pd.concat([fix_action, Efix_df], axis=1)
    fix_df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'subjectID', 'trialNum', 'onsettime',
                      'stimName', 'bid', 'RT', 'stimType', 'stimId', 'to_remove', 'E_timeStamp',
                      'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size', 'dominant_eye']
    fix_df.drop(['to_remove1', 'to_remove'], inplace=True, axis=1)
    fix_df.reset_index(drop=True, inplace=True)

    Esacc_df = df[df['timeStamp'].str.contains('ESACC', na=False)]
    Esacc_df.dropna(inplace=True, axis=1)
    sacc_action = Esacc_df["timeStamp"].str.split(expand=True)
    sacc_df = pd.concat([sacc_action, Esacc_df], axis=1)
    sacc_df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'subjectID', 'trialNum', 'onsettime',
                       'stimName', 'bid', 'RT', 'stimType', 'stimId', 'to_remove', 'E_timeStamp',
                       'duration', 'S_X_axis', 'S_Y_axis', 'E_X_axis', 'E_Y_axis', 'dominant_eye']
    sacc_df.drop(['to_remove1', 'to_remove'], inplace=True, axis=1)
    sacc_df.reset_index(drop=True, inplace=True)

    # List of all fields that will be updated for later use (fixation and saccade)
    df_fields = ['S_timeStamp', 'E_timeStamp', 'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size', 'S_X_axis',
                 'S_Y_axis', 'E_X_axis', 'E_Y_axis']

    # add 'sampleId' field for each uniqe sample
    fix_df['sampleId'] = fix_df['subjectID'].astype(str) + '_' + fix_df['stimName'].astype(str)
    # Removes data with no bid value
    fix_df = fix_df[fix_df.bid != 999]
    # For each field update relevant type and clean not relevant data
    for field in df_fields:
        if field in fix_df.columns:
            print(field)
            print(type(fix_df[field][0]))
            if type(fix_df[field][0]) == str:
                fix_df[field] = fix_df[field].str.strip()
            fix_df[field] = pd.to_numeric(fix_df[field], errors='coerce')
            fix_df[field] = fix_df[field].round()
            fix_df.dropna(inplace=True)
            fix_df[field] = fix_df[field].astype(int)
            fix_df.reset_index(drop=True, inplace=True)

    # add 'sampleId' field for each uniqe sample
    sacc_df['sampleId'] = sacc_df['subjectID'].astype(str) + '_' + sacc_df['stimName'].astype(str)
    # Removes data with no bid value
    sacc_df = sacc_df[sacc_df.bid != 999]
    # For each field update relevant type and clean not relevant data
    for field in df_fields:
        if field in sacc_df.columns:
            print(field)
            print(type(sacc_df[field][0]))
            if type(sacc_df[field][0]) == str:
                sacc_df[field] = sacc_df[field].str.strip()
            sacc_df[field] = pd.to_numeric(sacc_df[field], errors='coerce')
            sacc_df[field] = sacc_df[field].round()
            sacc_df.dropna(inplace=True)
            sacc_df[field] = sacc_df[field].astype(int)
            sacc_df.reset_index(drop=True, inplace=True)

    # In this section for each stim (snack or face) get the datapoints within the stim boundaries
    # For fixation dataset and for saccade dataset
    # TODO: Create one function that gets for input the stim type and size
    #  and outputs the DF with only datapoints within the stim region

    # FIXATION - stim is snack
    stim_id = 2
    stim_resolution = (snack_stim_size)
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    # get only datapoints within stim boundaries
    FIXstimARegionDataDf = fix_df[((fix_df['stimId'] == stim_id) & (fix_df['avg_X_axis'] >= min_x) & (fix_df['avg_X_axis'] <= max_x) &
                                   (fix_df['avg_Y_axis'] >= min_y) & (fix_df['avg_Y_axis'] <= max_y)) == True]

    # FIXATION - stim face
    stim_resolution = (face_stim_size)
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    #get only datapoints within stim boundaries
    FIXstimBRegionDataDf = fix_df[(((fix_df['stimId'] != stim_id) & (fix_df['avg_X_axis'] >= min_x) & (fix_df['avg_X_axis'] <= max_x) & (
            fix_df['avg_Y_axis'] >= min_y) & (fix_df['avg_Y_axis'] <= max_y)) == True)]

    FIXbyImgRegionDataDf = pd.concat([FIXstimARegionDataDf, FIXstimBRegionDataDf])
    fix_df_unique_after = FIXbyImgRegionDataDf.sampleId.nunique()
    FIXbyImgRegionDataDf.reset_index(drop=True, inplace=True)
    #FIXbyImgRegionDataDf.drop(FIXbyImgRegionDataDf.columns[[0]], axis=1, inplace=True)

    # SACCADE - stim is snack
    stim_id = 2
    stim_resolution = (snack_stim_size)
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    # get only datapoints within stim boundaries
    SACCstimARegionDataDf = sacc_df[((sacc_df['stimId'] == stim_id) & (sacc_df['S_X_axis'] >= min_x) & (sacc_df['S_X_axis'] <= max_x) &
                            (sacc_df['S_Y_axis'] >= min_y) & (sacc_df['S_Y_axis'] <= max_y) & (sacc_df['E_X_axis'] >= min_x) & (sacc_df['E_X_axis'] <= max_x) &
                            (sacc_df['E_Y_axis'] >= min_y) & (sacc_df['E_Y_axis'] <= max_y)) == True]

    # SACCADE - stim face
    stim_resolution = (face_stim_size)
    min_x, max_x, min_y, max_y = find_stim_boundaries(screen_resolution, stim_resolution)
    # get only datapoints within stim boundaries
    SACCstimBRegionDataDf = sacc_df[(((sacc_df['stimId'] != stim_id) & (sacc_df['S_X_axis'] >= min_x) & (sacc_df['S_X_axis'] <= max_x) & (
            sacc_df['S_Y_axis'] >= min_y) & (sacc_df['S_Y_axis'] <= max_y) & (sacc_df['E_X_axis'] >= min_x) & (sacc_df['E_X_axis'] <= max_x) & (
            sacc_df['E_Y_axis'] >= min_y) & (sacc_df['E_Y_axis'] <= max_y)) == True)]


    SACCbyImgRegionDataDf = pd.concat([SACCstimARegionDataDf, SACCstimBRegionDataDf])
    sacc_df_unique_after = SACCbyImgRegionDataDf.sampleId.nunique()
    SACCbyImgRegionDataDf.reset_index(drop=True, inplace=True)


    return FIXbyImgRegionDataDf, SACCbyImgRegionDataDf, fix_df_unique_after, sacc_df_unique_after

