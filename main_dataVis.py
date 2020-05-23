import sys
sys.path.append('../')
import pandas as pd
from modules.data.preprocessing import DataPreprocess
from modules.data.visualization import DataVis
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
import os
import yaml



def get_datasets(x_subjects):
    path = os.getcwd()
    expconfig = "/modules/config/experimentconfig.yaml"
    with open(path + expconfig, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
    stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
    stimArray = [stimFace, stimSnack]
    data = DataPreprocess(cfg['exp']['etp']['name'],
                          cfg['exp']['etp']['both_eye_path'],
                          cfg['exp']['etp']['one_eye_path1'],
                          cfg['exp']['etp']['trial_start_str'],
                          cfg['exp']['etp']['trial_end_str'],
                          cfg['exp']['etp']['output_file_both_eye'],
                          cfg['exp']['etp']['output_file_one_eye1'], [stimSnack, stimFace])
    fixation_only = False
    datasetbuilder = DatasetBuilder([stimSnack, stimFace])

    try:
        print("Log... reading fixation and scanpath df's")
        fixation_df = pd.read_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
        scanpath_df = pd.read_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
    except:
        try:
            print("Log... reading tidy df")
            tidy_data = pd.read_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
            fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
            scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
            fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
            scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
        except:
            try:
                print("Log... reading raw data csv")
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/" + x_subjects + "_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
            except:
                print("Log... processing raw data to csv")
                both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded(fixation_only)
                one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded(fixation_only)
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/" + x_subjects + "_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")

    return stimArray, scanpath_df, fixation_df


stimArray, scanpath_df_old, fixation_df_old = get_datasets("40")
stimArray, scanpath_df_new, fixation_df_new = get_datasets("new")
fixation_df = pd.concat([fixation_df_old, fixation_df_new])
scanpath_df = pd.concat([scanpath_df_old, scanpath_df_new])

datavis = DataVis('/etp_data/Stim_0/', '/etp_data/visualized_data/', stimArray, "Face")
datavis.visualize_for_speciffic_stim(fixation_df, scanpath_df)