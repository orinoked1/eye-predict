import numpy as np
import cv2
import os
from eye_tracking_data_parser import raw_data_preprocess as parser
from sklearn.utils import shuffle
import pandas as pd
import pickle
import glob

def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    y = os.getcwd()
    asc_files_path = y+'/raw_data/etp_pailot_data'
    txt_files_path = y+'/raw_data/etp_pailot_data'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "pailot_data.csv"


    scale_ranking_bmm_short_data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)





    return

# get_raw_data()
y = os.getcwd()
# read csv into DF
pailot_data_df = pd.read_csv(y + '/pailot_data.csv')
print('x')