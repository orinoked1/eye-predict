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
    asc_files_path = y+'/etp_data/Output_118_125'
    txt_files_path = y+'/etp_data/Output_118_125'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "output_data_both_eyes_118_125.csv"


    data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)





    return data_csv_path

path = get_raw_data()