import sys
sys.path.append('../')
import pandas as pd
from modules.data.preprocessing import DataPreprocess
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
import os
import yaml
from modules.data.visualization import DataVis


path = os.getcwd()
expconfig = "/modules/config/experimentconfig.yaml"
with open(path + expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

stimFood = Stim(cfg['exp']['weizmann']['stimFood']['name'], cfg['exp']['weizmann']['stimFood']['id'], cfg['exp']['weizmann']['stimFood']['size'])
data = DataPreprocess(cfg['exp']['weizmann']['name'],
                      cfg['exp']['weizmann']['both_eye_path'],
                      cfg['exp']['weizmann']['one_eye_path1'],
                      cfg['exp']['weizmann']['trial_start_str'],
                      cfg['exp']['weizmann']['trial_end_str'],
                      cfg['exp']['weizmann']['output_file_both_eye'],
                      cfg['exp']['weizmann']['output_file_one_eye1'], [stimFood])

fixation_only = False
#both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded(fixation_only)
one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded(fixation_only)

try:
    tidy_data = pd.read_pickle(path + "/weizmann/processed/tidy_pailot_data.pkl")
except:
    #both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
    one_eye_data = pd.read_csv(path + "/weizmann/processed/pailot_data.csv") #(one_eye_data_path)
    all_data = one_eye_data
    #all_data = pd.concat([both_eye_data, one_eye_data])
    tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
    tidy_data.to_pickle(path + "/weizmann/processed/tidy_pailot_data.pkl")

datasetbuilder = DatasetBuilder([stimFood])
fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
fixation_df.to_pickle(path + "/weizmann/processed/fixation_pailot_data.pkl")
scanpath_df.to_pickle(path + "/weizmann/processed/scanpath_pailot_data.pkl")

datavis = DataVis(cfg['exp']['weizmann']['stim_path'], cfg['exp']['weizmann']['visualization_path'], [stimFood], "full_run")

datavis.visualize(fixation_df, scanpath_df)