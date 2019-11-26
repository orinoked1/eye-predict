import visualize_data_functions as vis
import sys
sys.path.append('../')
import ds_readers as ds
import pandas as pd
import pickle
from modules.data.preprocessing import DataPreprocess
from modules.data.datasets import DatasetBuilder
from modules.data.visualization import DataVis
from modules.data.stim import Stim
import os
import yaml


path = os.getcwd()
expconfig = "/modules/config/experimentconfig.yaml"
with open(path + expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

stimSnack = Stim(cfg['exp']['test']['stimSnack']['name'], cfg['exp']['test']['stimSnack']['id'], cfg['exp']['test']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['test']['stimFace']['name'], cfg['exp']['test']['stimFace']['id'], cfg['exp']['test']['stimFace']['size'])
data = DataPreprocess(cfg['exp']['test']['both_eye_path'], cfg['exp']['test']['one_eye_path1'], cfg['exp']['test']['trial_start_str'],
                      cfg['exp']['test']['trial_end_str'], cfg['exp']['test']['output_file_both_eye'], cfg['exp']['test']['output_file_one_eye1'], [stimSnack, stimFace])

both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded()
one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded()
both_eye_data = pd.read_csv(both_eye_data_path)
one_eye_data = pd.read_csv(one_eye_data_path)
all_data = pd.concat([both_eye_data, one_eye_data])
tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['test']['screen_resolution'])

datasetbuilder = DatasetBuilder([stimSnack, stimFace])
fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)

datavis = DataVis(cfg['exp']['test']['stim_path'], cfg['exp']['test']['visualization_path'], [stimSnack, stimFace], 0)

datavis.visualize(fixation_df, scanpath_df)

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