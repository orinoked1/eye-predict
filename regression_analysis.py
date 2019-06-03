import os
import pandas as pd
import ds_readers as ds
import visualize_data_functions as vis
from eye_tracking_data_parser import raw_data_preprocess as parser
import pickle


def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    y = os.getcwd()
    asc_files_path = y+'/etp_data/raw_data'
    txt_files_path = y+'/etp_data/raw_data'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "raw_data_01.csv"


    data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)





    return data_csv_path

#path = get_raw_data()

y = os.getcwd()
# read csv into DF
raw_data_df = pd.read_csv(y + '/raw_data_01.csv')


"""
Visualization lalala
"""
"""    
fixation_dataset = ds.get_fixation_dataset(raw_data_df, ([1080, 1920]))
with open('etp_test_fixation_dataset.pkl', 'wb') as f:
    pickle.dump(fixation_dataset, f)
"""

with open('etp_test_fixation_dataset.pkl', 'rb') as f:
    print('Getting fixation dataset')
    fixation_dataset = pickle.load(f)
fixation_df = pd.DataFrame(fixation_dataset)
fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']
"""
scanpath_dataset = ds.get_scanpath_dataset(raw_data_df, ([1080, 1920]))
with open('etp_test_scanpath_dataset.pkl', 'wb') as f:
    pickle.dump(scanpath_dataset, f)
"""

with open('etp_test_scanpath_dataset.pkl', 'rb') as f:
    print('Getting scanpath dataset')
    scanpath_dataset = pickle.load(f)
scanpath_df = pd.DataFrame(scanpath_dataset)
scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[1])

print('x')