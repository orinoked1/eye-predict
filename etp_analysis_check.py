import os
from eye_tracking_data_parser import raw_data_preprocess as parser
import pandas as pd


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

print('x')