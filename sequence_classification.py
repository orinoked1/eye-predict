import os
import pandas as pd
import ds_readers as ds
import pickle


path = os.getcwd()

raw_data_df_101_117 = pd.read_csv(path + '/output_data_both_eyes_101_117.csv')
raw_data_df_118_125 = pd.read_csv(path + '/output_data_both_eyes_118_125.csv')
allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])

scanpath_dataset = ds.get_scanpath_dataset(allSubjectsData, ([1080, 1920]))
with open('etp_scanpath_dataset_101_125.pkl', 'wb') as f:
    pickle.dump(scanpath_dataset, f)

scanpath_df = pd.DataFrame(scanpath_dataset)

scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

