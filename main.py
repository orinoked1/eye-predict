from eye_tracking_data_parser import raw_data_parser as parser
import torch
import pandas as pd

use_cuda = torch.cuda.is_available()

"""
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
"""
bdm_bmm_short_data_csv_path = 'eye_tracking_data_parser/bdm_bmm_short_data_df.csv'
scale_ranking_bmm_short_data_csv_path = 'eye_tracking_data_parser/scale_ranking_bmm_short_data_df.csv'

#read csv into DF
bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)

#merge the to datasets to one
data_df = pd.concat([bdm_bmm_short_data_df,scale_ranking_bmm_short_data_df])

#get tidy data frame
data_df = parser.data_tidying(data_df)
print('done')