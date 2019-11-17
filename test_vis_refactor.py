import visualize_data_functions as vis
import sys
sys.path.append('../')
import ds_readers as ds
from eye_tracking_data_parser import raw_data_preprocess as parser
import pandas as pd
import pickle
from modules.data.preprocessing import DataPreprocess
from modules.data.stim import Stim
import os
import yaml


path = os.getcwd()
expconfig = "/modules/config/experimentconfig.yaml"
with open(path + expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
data = DataPreprocess(cfg['exp']['etp']['asc_files_path'], cfg['exp']['etp']['txt_files_path'], cfg['exp']['etp']['trial_start_str'],
                             cfg['exp']['etp']['trial_end_str'], cfg['exp']['etp']['output_file'], [stimSnack, stimFace])

raw_data = data.read_eyeTracking_data()

def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    path = os.getcwd()
    asc_files_path = path +'/my_data_test'
    txt_files_path = path +'/my_data_test'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "my_data_test_processed_data.csv"


    data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)

    return data_csv_path

#path = get_raw_data()
path = os.getcwd()

data_df = pd.read_csv(path + '/my_data_test_processed_data.csv')

try:
    fixation_dataset = pd.read_pickle(path + "/fixation_dataset_my_data_test_v2.pklסס")
except:
    fixation_dataset = ds.get_fixation_dataset(data_df, ([1080, 1920]))
    with open('fixation_dataset_my_data_test_v2.pkl', 'wb') as f:
        pickle.dump(fixation_dataset, f)

try:
    scanpath_dataset = pd.read_pickle(path + "/scanpath_dataset__my_data_test_v2.pklxx")
except:
    scanpath_dataset = ds.get_scanpath_dataset(data_df, ([1080, 1920]))
    with open('scanpath_dataset__my_data_test_v2.pkl', 'wb') as f:
        pickle.dump(scanpath_dataset, f)


fixation_df = pd.DataFrame(fixation_dataset)
fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']
scanpath_df = pd.DataFrame(scanpath_dataset)
scanpath_df.columns = ['subject_id', 'stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

#map = fixation_df.fixationMap[6]

stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[0])