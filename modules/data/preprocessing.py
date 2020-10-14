import pandas as pd
import codecs
import os
import glob


class DataPreprocess:

    def __init__(self, exp_name, both_eye_data_path, one_eye_data_path, trial_start_str, trial_end_str, output_file_both_eye, output_file_one_eye, stimarray):
        self.exp_name = exp_name
        self.both_eye_data_path = both_eye_data_path
        self.one_eye_data_path = one_eye_data_path
        self.trial_start_str = trial_start_str
        self.trial_end_str = trial_end_str
        self.output_file_both_eye = output_file_both_eye
        self.output_file_one_eye = output_file_one_eye
        self.stimarray = stimarray

    def read_eyeTracking_data_both_eye_recorded(self, fixation_saccad_data):
        flag = 0
        path = os.getcwd()
        # Set directory name that contains output directory with asc and txt files
        asc_directory = self.both_eye_data_path
        txt_directory = self.both_eye_data_path
        excluded_participents = pd.read_table(path + txt_directory + '//' + 'excluded_participents.txt')
        # Set string represents trail start data records
        indexStartStr = self.trial_start_str  # 'TrialStart'
        # Set string represents trail ends data records
        indexEndStr = self.trial_end_str  # 'ScaleStart'
        # Run over each Ascii file and open it
        for ascFile in glob.glob(path + asc_directory + '//*asc'):
            ascFileName = os.path.basename(ascFile)
            tempList = ascFileName.split('_')
            subjectIntId = tempList[0]
            id = int(subjectIntId)
            # exclude partcipents
            if id in excluded_participents.exclude.values:
                print('Excluded subjectId - ' + subjectIntId)
                continue
            # Hack for reading new participents data only #####
            if id < 169:
                print('Excluded subjectId - ' + subjectIntId)
                continue
            if id > 174:
                break
            ##################################################
            print('Log.....Getting Ascii file data - ' + ascFileName)
            ascFile = codecs.open(path + asc_directory + '//' + ascFileName, encoding='utf-8-sig')
            ascData = ascFile.readlines()
            # Split data to columns and get only relevnt ones
            ascDf = pd.DataFrame(ascData, columns=['data'])
            ascDf = pd.DataFrame([x.split('\t') for x in list(ascDf['data'])])
            ascDf = ascDf[[0, 1, 2, 3, 4, 5, 6]]
            print('Log.....Getting txt file data for subject id - ' + subjectIntId)
            # Read txt file of subject same subject as asc file
            txtFileName = os.path.basename(glob.glob(path + txt_directory + '//*' + subjectIntId + '_Scale' + '*txt')[0])
            txtData = pd.read_table(path + txt_directory + '//' + txtFileName)
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

            # get onlydominant eye data
            txtFilePersonalDataName = os.path.basename(
                glob.glob(path + txt_directory + '//*' + subjectIntId + '_personalDetails' + '*txt')[0])
            txtpersonalData = pd.read_table(path + txt_directory + '//' + txtFilePersonalDataName)
            dominant_eye = txtpersonalData['dominant eye (1-right, 2-left)'].values[0]
            if fixation_saccad_data:
                if dominant_eye == 1:
                    allTrialsData['dominant_eye'] = 'R'
                else:
                    allTrialsData['dominant_eye'] = 'L'
            else:
                if dominant_eye == 1:
                    allTrialsData = allTrialsData.drop([1, 2, 3], axis=1)
                    allTrialsData.rename(columns={4: 1, 5: 2, 6: 3}, inplace=True)
                    allTrialsData['dominant_eye'] = 'R'
                else:
                    allTrialsData = allTrialsData.drop([4, 5, 6], axis=1)
                    allTrialsData['dominant_eye'] = 'L'

            # Appending all data to one DataFrame
            if flag == 0:
                allSubjectsData = pd.DataFrame(columns=allTrialsData.columns)
                flag = 1
            allSubjectsData = pd.concat([allSubjectsData, allTrialsData])

        if fixation_saccad_data:
            #Rename columns
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType',
                                       'stimId', '0', '1', '2', '3', '4', '5', '6', 'dominant_eye']
        elif self.exp_name == 'weizmann':
            # Rename columns if needed
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                                       'runNum', 'timeStamp', 'X_axis', 'Y_axis', 'pupil_size', 'dominant_eye']
        else:
            # Rename columns if needed
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                                       'timeStamp', 'X_axis', 'Y_axis', 'pupil_size', 'dominant_eye']

        # Store all subjects data DF as CSV
        allSubjectsData.to_csv(path + self.output_file_both_eye)
        data_csv_path = path + self.output_file_both_eye
        # returns path for data csv file
        return data_csv_path

    def read_eyeTracking_data_one_eye_recorded(self, fixation_saccad_data):
        flag = 0
        path = os.getcwd()
        # Set directory name that contains output directory with asc and txt files
        asc_directory = self.one_eye_data_path
        txt_directory = self.one_eye_data_path
        excluded_participents = pd.read_table(path + txt_directory + '//' + 'excluded_participents.txt')
        # Set string represents trail start data records
        indexStartStr = self.trial_start_str  # 'TrialStart'
        # Set string represents trail ends data records
        indexEndStr = self.trial_end_str  # 'ScaleStart'
        # Run over each Ascii file and open it
        for ascFile in glob.glob(path + asc_directory + '//*asc'):
            ascFileName = os.path.basename(ascFile)
            tempList = ascFileName.split('_')
            subjectIntId = tempList[0]
            id = int(subjectIntId)
            if id in excluded_participents.exclude.values:
                print('Excluded subjectId - ' + subjectIntId)
                continue
            # Hack for reading new participents data only #####
            if id <= 128:
                print('Excluded subjectId - ' + subjectIntId)
                continue
            if id > 146:
                break
            ##################################################
            print('Log.....Getting Ascii file data - ' + ascFileName)
            ascFile = codecs.open(path + asc_directory + '//' + ascFileName, encoding='utf-8-sig')
            ascData = ascFile.readlines()
            # Split data to columns and get only relevnt ones
            ascDf = pd.DataFrame(ascData, columns=['data'])
            ascDf = pd.DataFrame([x.split('\t') for x in list(ascDf['data'])])
            if fixation_saccad_data:
                ascDf = ascDf[[0, 1, 2, 3, 4, 5, 6]]
            else:
                ascDf = ascDf[[0, 1, 2, 3]]
            print('Log.....Getting txt file data for subject id - ' + subjectIntId)
            # Read txt file of subject same subject as asc file
            txtFileName = os.path.basename(glob.glob(path + txt_directory + '//*' + subjectIntId + '_Scale' + '*txt')[0])
            txtData = pd.read_table(path + txt_directory + '//' + txtFileName)
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

            # add dominant eye field
            txtFilePersonalDataName = os.path.basename(
                glob.glob(path + txt_directory + '//*' + subjectIntId + '_personalDetails' + '*txt')[0])
            txtpersonalData = pd.read_table(path + txt_directory + '//' + txtFilePersonalDataName)
            dominant_eye = txtpersonalData['dominant eye (1-right, 2-left)'].values[0]
            if dominant_eye == 1:
                allTrialsData['dominant_eye'] = 'R'
            else:
                allTrialsData['dominant_eye'] = 'L'

            # Appending all data to one DataFrame
            if flag == 0:
                allSubjectsData = pd.DataFrame(columns=allTrialsData.columns)
                flag = 1
            allSubjectsData = pd.concat([allSubjectsData, allTrialsData])

        if fixation_saccad_data:
            # Rename columns if needed
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType',
                                       'stimId', '0', '1', '2', '3', '4',
                                       '5', '6', '7']
        elif self.exp_name == 'weizmann':
            # Rename columns if needed
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                                       'runNum', 'timeStamp', 'X_axis', 'Y_axis', 'pupil_size', 'dominant_eye']
        else:
            # Rename columns if needed
            allSubjectsData.columns = ['subjectID', 'trialNum', 'onsettime', 'stimName', 'bid', 'RT', 'stimType', 'stimId',
                                       'timeStamp', 'X_axis', 'Y_axis', 'pupil_size', 'dominant_eye']

        # Store all subjects data DF as CSV
        allSubjectsData.to_csv(path + self.output_file_one_eye)
        data_csv_path = path + self.output_file_one_eye
        # returns path for data csv file
        return data_csv_path

    def find_stim_boundaries(self, screen_resolution, stim_resolution):
        min_x = (screen_resolution[1] / 2) - (stim_resolution[0] / 2)
        max_x = (screen_resolution[1] / 2) + (stim_resolution[0] / 2)
        min_y = (screen_resolution[0] / 2) - (stim_resolution[1] / 2)
        max_y = (screen_resolution[0] / 2) + (stim_resolution[1] / 2)

        return min_x, max_x, min_y, max_y

    def data_tidying_for_dataset_building(self, df, screen_resolution):
        print('Log..... Data tidying')
        flag = 0
        screen_resolution = [int(x) for x in screen_resolution.split(',')]
        # add 'sampleId' field for each uniqe sample
        df['sampleId'] = df['subjectID'].astype(str) + '_' + df['stimName'].astype(str)

        # Change X, Y and timeStamp data from String to Numeric changing strings " . ", "EBLINK", FIX", "SACC" to nan
        df.X_axis = pd.to_numeric(df.X_axis, errors='coerce')
        df.Y_axis = pd.to_numeric(df.Y_axis, errors='coerce')
        df.timeStamp = pd.to_numeric(df.timeStamp, errors='coerce')
        df.X_axis = df.X_axis.round()
        df.Y_axis = df.Y_axis.round()
        # Remove Nan
        df.dropna(inplace=True)
        df.X_axis = df.X_axis.astype(int)
        df.Y_axis = df.Y_axis.astype(int)
        df.timeStamp = df.timeStamp.astype(int)
        df = df[df.bid != 999]
        df.reset_index(drop=True, inplace=True)
        for stim in self.stimarray:
            stim_id = int(stim.id)
            stim_resolution = stim.size
            min_x, max_x, min_y, max_y = self.find_stim_boundaries(screen_resolution, stim_resolution)
            # get only datapoints within stim boundaries
            stimRegion = df[((df['stimId'] == stim_id) & (df['X_axis'] >= min_x) & (df['X_axis'] <= max_x) &
                                    (df['Y_axis'] >= min_y) & (df['Y_axis'] <= max_y)) == True]
            # Shifting x,y datapoint to start from (0,0) point
            stimRegion.X_axis = stimRegion.X_axis - min_x
            stimRegion.Y_axis = stimRegion.Y_axis - min_y

            # Appending all stims to one DataFrame
            if flag == 0:
                byRegionDf = pd.DataFrame(columns=stimRegion.columns)
                flag = 1
            byRegionDf = pd.concat([byRegionDf, stimRegion])

        byRegionDf.reset_index(drop=True, inplace=True)
        byRegionDf.drop(byRegionDf.columns[[0]], axis=1, inplace=True)

        return byRegionDf

    def data_tidying_for_fixation_dataset_building(self, df, screen_resolution):
        print('Log..... Data tidying')
        flag = 0
        screen_resolution = [int(x) for x in screen_resolution.split(',')]
        # Get relevant raw data for fixations and saccades
        df = df[df['action'].str.contains('EFIX', na=False)]
        df.dropna(inplace=True, axis=1)
        fix_action = df["action"].str.split(expand=True)
        df = pd.concat([fix_action, df], axis=1)
        df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'subjectID', 'trialNum', 'onsettime',
                          'stimName', 'bid', 'RT', 'stimType', 'stimId', 'to_remove', 'E_timeStamp',
                          'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size', 'dominant_eye']
        df.drop(['to_remove1', 'to_remove'], inplace=True, axis=1)
        df = df[df['eye'] == df['dominant_eye']]
        df.reset_index(drop=True, inplace=True)


        # List of all fields that will be updated for later use (fixation and saccade)
        df_fields = ['timeStamp', 'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size']

        # add 'sampleId' field for each uniqe sample
        df['sampleId'] = df['subjectID'].astype(str) + '_' + df['stimName'].astype(str)
        # Removes data with no bid value
        fix_df = df[df.bid != 999]
        # For each field update relevant type and clean not relevant data
        for field in df_fields:
            if field in df.columns:
                print(field)
                print(type(df[field][0]))
                if type(df[field][0]) == str:
                    df[field] = df[field].str.strip()
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    df[field] = df[field].round()
                    df.dropna(inplace=True)
                    df[field] = df[field].astype(int)
                    df.reset_index(drop=True, inplace=True)


        # In this section for each stim get the datapoints within the stim boundaries
        for stim in self.stimarray:
            stim_id = stim.id
            stim_resolution = (stim.size)
            min_x, max_x, min_y, max_y = self.find_stim_boundaries(screen_resolution, stim_resolution)
            # get only datapoints within stim boundaries
            stimRegion = df[
                ((df['stimId'] == stim_id) & (df['avg_X_axis'] >= min_x) & (df['avg_X_axis'] <= max_x) &
                 (df['avg_Y_axis'] >= min_y) & (df['avg_Y_axis'] <= max_y)) == True]

            # Shifting x,y datapoint to start from (0,0) point
            stimRegion.X_axis = stimRegion.X_axis - min_x
            stimRegion.Y_axis = stimRegion.Y_axis - min_y

            # Appending all stims to one DataFrame
            if flag == 0:
                byRegionDf = pd.DataFrame(columns=stimRegion.columns)
                flag = 1
            byRegionDf = pd.concat([byRegionDf, stimRegion])

        byRegionDf.reset_index(drop=True, inplace=True)
        byRegionDf.drop(byRegionDf.columns[[0]], axis=1, inplace=True)

        return byRegionDf

    def data_tidying_for_analysis_fix_sacc(self, df, screen_resolution):
        print('Log..... Data tidying')
        screen_resolution = [int(x) for x in screen_resolution.split(',')]
        # Get relevant raw data for fixations and saccades
        Efix_df = df[df['0'].str.contains('EFIX', na=False)]

        Efix_df.dropna(inplace=True, axis=1)
        fix_action = Efix_df["0"].str.split(expand=True)
        fix_df = pd.concat([fix_action, Efix_df], axis=1)
        """
        fix_df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'subjectID',
                          'trialNum', 'onsettime', 'stimName',
                          'bid', 'RT', 'stimType', 'stimId', 'to_remove',  'E_timeStamp',
                          'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size',
                          'dominant_eye']
        """
        fix_df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'E_timeStamp',
                          'duration', 'avg_X_axis', 'avg_Y_axis', 'avg_pupil_size',
                          'RT', 'to_remove', 'bid', 'onsettime', 'stimId', 'stimName',
                          'stimType',  'subjectID', 'trialNum']

        fix_df.drop(['to_remove1', 'to_remove'], inplace=True, axis=1)
        fix_df.reset_index(drop=True, inplace=True)

        Esacc_df = df[df['0'].str.contains('ESACC', na=False)]
        Esacc_df.dropna(inplace=True, axis=1)
        sacc_action = Esacc_df["0"].str.split(expand=True)
        sacc_df = pd.concat([sacc_action, Esacc_df], axis=1)
        sacc_df.columns = ['action', 'eye', 'S_timeStamp', 'to_remove1', 'E_timeStamp',
                           'duration', 'S_X_axis', 'S_Y_axis', 'E_X_axis', 'E_Y_axis',
                           'RT', 'to_remove', 'bid', 'onsettime', 'stimId', 'stimName',
                           'stimType', 'subjectID', 'trialNum']
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
                na1 = fix_df[fix_df.isnull().any(axis=1)]
                print(field)
                print(type(fix_df[field][0]))
                #if type(fix_df[field][0]) == str:
                #    fix_df[field] = fix_df[field].str.strip()
                fix_df[field] = pd.to_numeric(fix_df[field], errors='coerce')
                fix_df[field] = fix_df[field].round()
                na = fix_df[fix_df.isnull().any(axis=1)]
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
                #if type(sacc_df[field][0]) == str:
                #    sacc_df[field] = sacc_df[field].str.strip()
                sacc_df[field] = pd.to_numeric(sacc_df[field], errors='coerce')
                sacc_df[field] = sacc_df[field].round()
                sacc_df.dropna(inplace=True)
                sacc_df[field] = sacc_df[field].astype(int)
                sacc_df.reset_index(drop=True, inplace=True)


        # In this section for each stim get the datapoints within the stim boundaries
        # For fixation dataset
        flag = 0
        for stim in self.stimarray:
            stim_id = stim.id
            stim_resolution = (stim.size)
            min_x, max_x, min_y, max_y = self.find_stim_boundaries(screen_resolution, stim_resolution)
            # get only datapoints within stim boundaries
            stimRegion_fix_df = fix_df[
                ((fix_df['stimId'] == stim_id) & (fix_df['avg_X_axis'] >= min_x) & (fix_df['avg_X_axis'] <= max_x) &
                 (fix_df['avg_Y_axis'] >= min_y) & (fix_df['avg_Y_axis'] <= max_y)) == True]

            # Shifting x,y datapoint to start from (0,0) point
            #stimRegion_fix_df.X_axis = stimRegion_fix_df.X_axis - min_x
            #byRegion_fix_Df.Y_axis = stimRegion_fix_df.Y_axis - min_y

            # Appending all stims to one DataFrame
            if flag == 0:
                byRegion_fix_Df = pd.DataFrame(columns=stimRegion_fix_df.columns)
                flag = 1
            byRegion_fix_Df = pd.concat([byRegion_fix_Df, stimRegion_fix_df])

        byRegion_fix_Df.reset_index(drop=True, inplace=True)
        byRegion_fix_Df.drop(byRegion_fix_Df.columns[[0]], axis=1, inplace=True)

        #for saccade dataset
        flag = 0
        for stim in self.stimarray:
            stim_id = stim.id
            stim_resolution = (stim.size)
            min_x, max_x, min_y, max_y = self.find_stim_boundaries(screen_resolution, stim_resolution)
            # get only datapoints within stim boundaries
            stimRegion_sacc_df = sacc_df[
                ((sacc_df['stimId'] == stim_id) & (sacc_df['S_X_axis'] >= min_x) & (sacc_df['S_X_axis'] <= max_x) &
                 (sacc_df['S_Y_axis'] >= min_y) & (sacc_df['S_Y_axis'] <= max_y) & (sacc_df['E_X_axis'] >= min_x) & (
                         sacc_df['E_X_axis'] <= max_x) &
                 (sacc_df['E_Y_axis'] >= min_y) & (sacc_df['E_Y_axis'] <= max_y)) == True]
            # Appending all stims to one DataFrame
            if flag == 0:
                byRegion_sacc_Df = pd.DataFrame(columns=stimRegion_sacc_df.columns)
                flag = 1
            byRegion_sacc_Df = pd.concat([byRegion_sacc_Df, stimRegion_sacc_df])

        byRegion_sacc_Df.reset_index(drop=True, inplace=True)
        byRegion_sacc_Df.drop(byRegion_sacc_Df.columns[[0]], axis=1, inplace=True)

        return byRegion_fix_Df, byRegion_sacc_Df

    def participents_data(self):
        flag = 0
        path = os.getcwd()
        # Set directory name that contains output directory with asc and txt files
        asc_directory = '/etp_data/Output'
        txt_directory = '/etp_data/Output'
        path = '/export/home/DATA/schonberglab/pycharm_eyePredict/'
        excluded_participents = pd.read_table(path + txt_directory + '/' + 'excluded_participents.txt')
        # Set string represents trail start data records
        indexStartStr = self.trial_start_str  # 'TrialStart'
        # Set string represents trail ends data records
        indexEndStr = self.trial_end_str  # 'ScaleStart'
        # Run over each Ascii file and open it
        for ascFile in glob.glob(path + asc_directory + '//*personalDetails*'):
            ascFileName = os.path.basename(ascFile)
            tempList = ascFileName.split('_')
            subjectIntId = tempList[0]
            id = int(subjectIntId)
            # exclude partcipents
            #if id in excluded_participents.exclude.values:
            #    print('Excluded subjectId - ' + subjectIntId)
            #    continue
            # get participent personal details
            txtFilePersonalDataName = os.path.basename(
                glob.glob(path + txt_directory + '//*' + subjectIntId + '_personalDetails' + '*txt')[0])
            txtpersonalData = pd.read_table(path + txt_directory + '//' + txtFilePersonalDataName)

            # Appending all data to one DataFrame
            if flag == 0:
                allSubjectsPersonalData = pd.DataFrame(columns=txtpersonalData.columns)
                flag = 1
            allSubjectsPersonalData = pd.concat([allSubjectsPersonalData, txtpersonalData])

        # Store all subjects data DF as CSV
        allSubjectsPersonalData.to_csv(path + 'etp_data/processed/' + 'participents_personal_data.csv')
        print('p')
