from eye_tracking_data_parser import raw_data_preprocess as parser
import get_data_functions as get
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()

def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    asc_files_path = '../raw_data/scale_ranking_bmm_short_data/output/asc'
    txt_files_path = '../raw_data/scale_ranking_bmm_short_data/output/txt'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "scale_ranking_bmm_short_data.csv"
    scale_ranking_bmm_short_data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)

    #read 'bdm_bmm_short_data' row data into csv
    #TODO: read info from config file
    asc_files_path = '../raw_data/bdm_bmm_short_data/output/asc'
    txt_files_path = '../raw_data/bdm_bmm_short_data/output/txt'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'Response'
    csv_file_name = "bdm_bmm_short_data.csv"
    bdm_bmm_short_data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)


    # read csv into DF
    bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
    scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)

    return  bdm_bmm_short_data_df, scale_ranking_bmm_short_data_df



bdm_bmm_short_data_csv_path = 'eye_tracking_data_parser/bdm_bmm_short_data_df.csv'
scale_ranking_bmm_short_data_csv_path = 'eye_tracking_data_parser/scale_ranking_bmm_short_data_df.csv'

try:
    # read csv into DF
    bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
    scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)
except:
    print('ERROR: csv files does not exist! -> start running def to create it')
    bdm_bmm_short_data_df, scale_ranking_bmm_short_data_df  = get_raw_data()

#merge the to datasets to one
data_df = pd.concat([bdm_bmm_short_data_df,scale_ranking_bmm_short_data_df])


try:
   with open('fixation_dataset_v1.pkl', 'rb') as f:
       fixation_dataset = pickle.load(f)
except:
   print('ERROR: fixation dataset does not exist! -> start running def to create it')
   fixation_dataset = get.get_fixation_dataset(data_df, ([1080, 1920]))
   with open('fixation_dataset_v1.pkl', 'wb') as f:
       pickle.dump(fixation_dataset, f)


try:
    with open('scanpath_dataset.pkl', 'rb') as f:
        scanpath_dataset = pickle.load(f)
except:
    print('ERROR: Scanpath dataset does not exist! -> start running def to create it')
    scanpath_dataset = get.get_scanpath_dataset(data_df, ([1080, 1920]))
    with open('scanpath_dataset.pkl', 'wb') as f:
        pickle.dump(scanpath_dataset, f)

print('x')

