import os
import pandas as pd
import ds_readers as ds
import visualize_data_functions as vis
from eye_tracking_data_parser import raw_data_preprocess as parser
import pickle
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


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

#path = get_raw_data()


y = os.getcwd()
# read csv into DF
raw_data_df = pd.read_csv(y + '/output_data_both_eyes_101_117.csv')
fix_df, sacc_df = parser.data_tidying_for_analysis(raw_data_df, [1080, 1920])

fix_df_calc = fix_df[fix_df['eye'] == 'R']
fix_df_calc.reset_index(drop=True, inplace=True)
fix_df_calc['fix_count'] = fix_df_calc.groupby('sampleId')['sampleId'].transform('count')
fix_df_calc['avg_fix_duration'] = fix_df_calc.groupby('sampleId')['duration'].transform('mean')
first_fix_data = fix_df_calc.groupby('sampleId').first().reset_index()
first_fix_corr = first_fix_data.corr(method ='pearson')
last_fix_data = fix_df_calc.groupby('sampleId').nth(-1).reset_index()
last_fix_corr = last_fix_data.corr(method ='pearson')

fix_corr_df = fix_df_calc.corr(method ='pearson')

sacc_df_calc = sacc_df[sacc_df['eye'] == 'R']
sacc_df_calc.reset_index(drop=True, inplace=True)
sacc_df_calc['sacc_count'] = sacc_df_calc.groupby('sampleId')['sampleId'].transform('count')
sacc_df_calc['avg_sacc_duration'] = sacc_df_calc.groupby('sampleId')['duration'].transform('mean')
sacc_df_calc['X_diff'] = sacc_df_calc['E_X_axis'] - sacc_df_calc['S_X_axis']
sacc_df_calc['Y_diff'] = sacc_df_calc['E_Y_axis'] - sacc_df_calc['S_Y_axis']
sacc_df_calc['X_diff'] = sacc_df_calc['X_diff'].abs()
sacc_df_calc['Y_diff'] = sacc_df_calc['Y_diff'].abs()
sacc_df_calc['avg_sacc_X_diff'] = sacc_df_calc.groupby('sampleId')['X_diff'].transform('mean')
sacc_df_calc['avg_sacc_Y_diff'] = sacc_df_calc.groupby('sampleId')['Y_diff'].transform('mean')
first_sacc_data = sacc_df_calc.groupby('sampleId').first().reset_index()
first_sacc_corr = first_sacc_data.corr(method ='pearson')
last_sacc_data = sacc_df_calc.groupby('sampleId').nth(-1).reset_index()
last_sacc_corr = last_sacc_data.corr(method ='pearson')

sacc_corr_df = sacc_df_calc.corr(method ='pearson')

ax = sns.scatterplot(x="bid", y="avg_sacc_X_diff", data=first_sacc_data)
ax.set_title('Bid - Fix count Correlation')
plt.show()

print('x')

"""
##### Visualization lalala

 
#fixation_dataset = ds.get_fixation_dataset(raw_data_df, ([1080, 1920]))
#with open('etp_test_fixation_dataset.pkl', 'wb') as f:
#    pickle.dump(fixation_dataset, f)


with open('etp_test_fixation_dataset.pkl', 'rb') as f:
    print('Getting fixation dataset')
    fixation_dataset = pickle.load(f)
fixation_df = pd.DataFrame(fixation_dataset)
fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']

#scanpath_dataset = ds.get_scanpath_dataset(raw_data_df, ([1080, 1920]))
#with open('etp_test_scanpath_dataset.pkl', 'wb') as f:
#    pickle.dump(scanpath_dataset, f)


with open('etp_test_scanpath_dataset.pkl', 'rb') as f:
    print('Getting scanpath dataset')
    scanpath_dataset = pickle.load(f)
scanpath_df = pd.DataFrame(scanpath_dataset)
scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[1])


"""