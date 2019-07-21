import os
import pandas as pd
from eye_tracking_data_parser import eyeTracker_data_preprocess as parser
import statsmodels.api as sm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    path = os.getcwd()
    asc_files_path = path +'/CAT_MRI_faces_data'
    txt_files_path = path +'/CAT_MRI_faces_data'
    trial_satart_str = 'trial'
    trial_end_str = 'fixation cross'
    csv_file_name = "CAT_MRI_faces_data.csv"


    data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)





    return data_csv_path

path = get_raw_data()


path = os.getcwd()
# read csv into DF
"""
raw_data_df_101_117 = pd.read_csv(y + '/output_data_both_eyes_101_117.csv')
raw_data_df_118_125 = pd.read_csv(y + '/output_data_both_eyes_118_125.csv')
allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])
fix_df, sacc_df, fix_N, sacc_N = parser.data_tidying_for_analysis(allSubjectsData, [1080, 1920])

fix_df.to_pickle("fix_df.pkl")
sacc_df.to_pickle("sacc_df.pkl")
"""

fix_df = pd.read_pickle(path +"/fix_df.pkl")
sacc_df = pd.read_pickle(path + "/sacc_df.pkl")

fix_df_calc = fix_df[fix_df['eye'] == 'R']
fix_df_calc.reset_index(drop=True, inplace=True)
fix_df_calc['fix_count'] = fix_df_calc.groupby('sampleId')['sampleId'].transform('count')
fix_df_calc['avg_fix_duration'] = fix_df_calc.groupby('sampleId')['duration'].transform('mean')
first_fix_data = fix_df_calc.groupby('sampleId').first().reset_index()
first_fix_data.to_csv('fixation_data.csv')

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
    """
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


    """
    # Load the diabetes dataset
    #diabetes = df

    # Use only one feature
    diabetes_X = df

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = target[:-20]
    diabetes_y_test = target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    print('Score fanc - Variance score: %.2f' % regr.score(diabetes_X, target))

    # Plot outputs
    ax1 = sns.distplot(diabetes_y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(diabetes_y_pred, hist=False, color="b", label="Fitted Values", ax=ax1)

    #plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    #plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()

    return


# Load the data (fixation or saccade)
fixation_data = pd.read_csv(y + "/fixation_data.csv")
saccade_data = pd.read_csv(y + "/saccade_data.csv")
subjects = range(102,126)
for subjectId in subjects:
    # stim type - Face or Snack
    oneSubject_oneStimID_multipleLinear_regression(subjectId, fixation_data, saccade_data, 'Face')


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