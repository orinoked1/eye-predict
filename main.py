import sys
sys.path.append('../')
import pandas as pd
from modules.data.preprocessing import DataPreprocess
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
from modules.models.cnn_lstm_mixed_training import CnnLstm
import os
import yaml


path = os.getcwd()
expconfig = "/modules/config/experimentconfig.yaml"
with open(path + expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
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
    fixation_df = pd.read_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
    scanpath_df = pd.read_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
except:
    try:
        print("Log... reading tidy df")
        tidy_data = pd.read_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
        fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
        scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
        fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
        scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
    except:
        try:
            print("Log... reading raw data csv")
            both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
            one_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_one_eye_data.csv") #(one_eye_data_path)
            all_data = pd.concat([both_eye_data, one_eye_data])
            tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
            tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
            fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
            scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
            fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
            scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
        except:
            print("Log... processing raw data to csv")
            both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded(fixation_only)
            one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded(fixation_only)
            both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
            one_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_one_eye_data.csv") #(one_eye_data_path)
            all_data = pd.concat([both_eye_data, one_eye_data])
            tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
            tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
            fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
            scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
            fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
            scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")

seed = 33
stimType = "Face"
patch_size = 60
saliency=False
is_patch = True
run_name = "_cnn_lstm_run_1_" + stimType
datasetbuilder = DatasetBuilder([stimSnack, stimFace])
scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(scanpath_df, stimType)
df = datasetbuilder.create_patches_dataset(scanpaths, images, labels, patch_size, saliency)
split_dataset = datasetbuilder.train_test_val_split_subjects_balnced(df, seed, is_patch)

cnn_lstm = CnnLstm(seed, split_dataset, saliency, patch_size, run_name)