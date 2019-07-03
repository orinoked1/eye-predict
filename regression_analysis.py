import os
import pandas as pd
import ds_readers as ds
import visualize_data_functions as vis
from eye_tracking_data_parser import raw_data_preprocess as parser
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


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
"""
raw_data_df_101_117 = pd.read_csv(y + '/output_data_both_eyes_101_117.csv')
raw_data_df_118_125 = pd.read_csv(y + '/output_data_both_eyes_118_125.csv')
allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])
fix_df, sacc_df, fix_N, sacc_N = parser.data_tidying_for_analysis(allSubjectsData, [1080, 1920])

fix_df.to_pickle("fix_df.pkl")
sacc_df.to_pickle("sacc_df.pkl")
"""
fix_df = pd.read_pickle(y +"/fix_df.pkl")
sacc_df = pd.read_pickle(y + "/sacc_df.pkl")


fix_df_calc = fix_df[fix_df['eye'] == 'R']
fix_df_calc.reset_index(drop=True, inplace=True)
fix_df_calc['fix_count'] = fix_df_calc.groupby('sampleId')['sampleId'].transform('count')
fix_df_calc['avg_fix_duration'] = fix_df_calc.groupby('sampleId')['duration'].transform('mean')
first_fix_data = fix_df_calc.groupby('sampleId').first().reset_index()
fix_corr = first_fix_data.groupby(['subjectID', 'stimId']).corr(method='spearman')
fix_corr.to_csv('fixations_correlation_by_subject_stimType.csv')
#first_fix_corr = first_fix_data.corr(method ='pearson')
#last_fix_data = fix_df_calc.groupby('sampleId').nth(-1).reset_index()
#last_fix_corr = last_fix_data.corr(method ='pearson')
#fix_corr_df = fix_df_calc.corr(method ='pearson')
#fix_corr_df.to_csv('fixations_correlations.csv')

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
sacc_corr = first_sacc_data.groupby(['subjectID', 'stimId']).corr(method='spearman')
sacc_corr.to_csv('saccade_correlation_by_subject_stimType.csv')



#x = fixdata['bid']
#plt.hist(x)
#plt.show()

# MLM fix
"""
fixdata = first_fix_data[first_fix_data['stimId'] == 1]
fixdata = fixdata[['bid', 'fix_count', 'subjectID']]
endog = fixdata["bid"]
fixdata["Intercept"] = 1
exog = fixdata[["Intercept", "fix_count"]]
md = sm.MixedLM(endog, exog, groups=fixdata["subjectID"], exog_re=exog)
mdf = md.fit()
print(mdf.summary())
"""
# MLM fix
saccdata = first_sacc_data[first_sacc_data['stimId'] == 1]
saccdata = saccdata[['bid', 'avg_sacc_X_diff', 'subjectID']]
endog = saccdata["bid"]
saccdata["Intercept"] = 1
exog = saccdata[["Intercept", "avg_sacc_X_diff"]]
md = sm.MixedLM(endog, exog, groups=saccdata["subjectID"], exog_re=exog)
mdf = md.fit()
print(mdf.summary())


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