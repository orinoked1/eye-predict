import os
import pandas as pd
from eye_tracking_data_parser import raw_data_preprocess as parser
import statsmodels.api as sm
from sklearn import linear_model


def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    path = os.getcwd()
    asc_files_path = path +'/etp_data/Output'
    txt_files_path = path +'/etp_data/Output'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "etp_processed_data.csv"


    data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)





    return data_csv_path

path = get_raw_data()


path = os.getcwd()
# Run this only once so you wont need to tide the data again (tidying takes time)
# read csv into DF
data = pd.read_csv(path + 'etp_processed_data.csv') # This is only for example the in no file with this name
fix_df, sacc_df, fix_N, sacc_N = parser.data_tidying_for_analysis(data, [1080, 1920], [520,690], [600, 480])
fix_df.to_pickle("fix_df.pkl")
sacc_df.to_pickle("sacc_df.pkl")

# One you run the above mark it as comment and just read the tidy pickle into a DF
fix_df = pd.read_pickle(path +"fix_df.pkl")
sacc_df = pd.read_pickle(path + "sacc_df.pkl")

# Add the relevant calculation for fixation DF and save as CSV file
fix_df_calc = fix_df[fix_df['eye'] == 'R']
fix_df_calc.reset_index(drop=True, inplace=True)
fix_df_calc['fix_count'] = fix_df_calc.groupby('sampleId')['sampleId'].transform('count')
fix_df_calc['avg_fix_duration'] = fix_df_calc.groupby('sampleId')['duration'].transform('mean')
first_fix_data = fix_df_calc.groupby('sampleId').first().reset_index()
first_fix_data.to_csv('fixation_data.csv')

# Add the relevant calculation for saccade DF and save as CSV file
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
first_sacc_data.to_csv('saccade_data.csv')

# The above code will result getting the fixation and saccade dataframes that can be used in further analysis


# Some more functions for analysis:

def spearman_correlations_grouped_by_subjectIDstimId(dataDF, name):
    corr = dataDF.groupby(['subjectID', 'stimId']).corr(method='spearman')
    corr.to_csv(name + '_correlation_by_subject_stimType.csv')
    """
    ########### example ############
    fix_corr = first_fix_data.groupby(['subjectID', 'stimId']).corr(method='spearman')
    fix_corr.to_csv('fixations_correlation_by_subject_stimType.csv')
    """

def mixedLM (dataDF, endog_field_name, exdog_field_name, group_name):
    data = dataDF
    endog = data[endog_field_name]
    data["Intercept"] = 1
    exog = data[["Intercept", exdog_field_name]]
    md = sm.MixedLM(endog, exog, groups=data[group_name], exog_re=exog)
    mdf = md.fit()
    print(mdf.summary())

    """
    ############ example ##################
    fixdata = first_fix_data[first_fix_data['stimId'] == 1]
    fixdata = fixdata[['bid', 'fix_count', 'subjectID']]
    endog = fixdata["bid"]
    fixdata["Intercept"] = 1
    exog = fixdata[["Intercept", "fix_count"]]
    md = sm.MixedLM(endog, exog, groups=fixdata["subjectID"], exog_re=exog)
    mdf = md.fit()
    print(mdf.summary())
    #########################################
    """

def oneSubject_oneStimID_multipleLinear_regression(subjectID, fix_dataDF, sacc_dataDF, stimType):
    fixData = fix_dataDF[['sampleId', 'subjectID', 'stimId', 'bid', 'RT', 'fix_count', 'avg_fix_duration']]
    saccData = sacc_dataDF[['sampleId', 'subjectID', 'stimId', 'bid', 'RT', 'sacc_count', 'avg_sacc_duration',
                            'avg_sacc_X_diff', 'avg_sacc_Y_diff']]

    fixData = fixData[fixData['subjectID'] == subjectID]
    if fixData.empty:
        return
    saccData = saccData[saccData['subjectID'] == subjectID]
    dataDF  = pd.concat([fixData.set_index('sampleId'),saccData.set_index('sampleId')], axis=1, join='inner').reset_index()
    dataDF = dataDF.loc[:,~dataDF.columns.duplicated()]
    faceStimDF = dataDF[dataDF['stimId'] == 1]
    snackStimDF = dataDF[dataDF['stimId'] == 2].reset_index()


    # data frame that contains the independent variables (marked as “df”)
    # and the data frame with the dependent variable (marked as “target”)
    if stimType == 'Face':
        df = faceStimDF[['RT', 'fix_count', 'avg_fix_duration', 'sacc_count', 'avg_sacc_duration',
                                'avg_sacc_X_diff', 'avg_sacc_Y_diff']]
        target = faceStimDF.bid
    else:
        df = snackStimDF[['RT', 'fix_count', 'avg_fix_duration', 'sacc_count', 'avg_sacc_duration',
                         'avg_sacc_X_diff', 'avg_sacc_Y_diff']]
        target = snackStimDF.bid

    X = df
    y = target
    #fit model
    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)
    #predictions
    #predictions = lm.predict(X)
    #print(predictions[0:5])
    #score

    print('Score for subjectId - ', subjectId)
    print(lm.score(X, y))

    return


# Load the data (fixation or saccade)
fixation_data = pd.read_csv(y + "/fixation_data.csv")
saccade_data = pd.read_csv(y + "/saccade_data.csv")
subjects = range(102,126)
for subjectId in subjects:
    # stim type - Face or Snack
    oneSubject_oneStimID_multipleLinear_regression(subjectId, fixation_data, saccade_data, 'Face')
