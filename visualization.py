import visualize_data_functions as vis
import pickle
import pandas as pd
import numpy as np
from eye_tracking_data_parser import raw_data_preprocess as parser


bdm_bmm_short_data_csv_path = '../eye_tracking_data_parser/bdm_bmm_short_data_df.csv'
scale_ranking_bmm_short_data_csv_path = '../eye_tracking_data_parser/scale_ranking_bmm_short_data_df.csv'


# read csv into DF
bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)
#merge the to datasets to one
data_df = pd.concat([bdm_bmm_short_data_df,scale_ranking_bmm_short_data_df])

with open('../fixation_dataset_v1.pkl', 'rb') as f:
    fixation_dataset = pickle.load(f)

with open('../scanpath_dataset.pkl', 'rb') as f:
    scanpath_dataset = pickle.load(f)


stimType = fixation_dataset[3131][1]
if stimType == 'Snacks':
    path = '/bdm_bmm_short_data/stim/'
else:
    path = '/scale_ranking_bmm_short_data/stim/'

fixation_df = pd.DataFrame(fixation_dataset)
fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']

scanpath_df = pd.DataFrame(scanpath_dataset)
scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']



#vis.map(fixation_dataset[3131][3], path, fixation_dataset[3131][0])

#vis.scanpath(scanpath_dataset[3131][3], path, scanpath_dataset[3131][0], False)

vis.map(fixation_dataset[3131][3], path, fixation_dataset[3131][0])

vis.scanpath(scanpath_dataset[3131][3], path, scanpath_dataset[3131][0], False)






