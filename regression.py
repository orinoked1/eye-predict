
import pandas as pd
from sklearn import linear_model
import os
import matplotlib.pyplot as plt


def oneSubject_oneStimID_multipleLinear_regression(subjectID, fix_dataDF, sacc_dataDF, stimType):
    fixData = fix_dataDF[['sampleId', 'subjectID', 'stimId', 'bid', 'RT', 'fix_count', 'avg_fix_duration']]
    saccData = sacc_dataDF[['sampleId', 'subjectID', 'stimId', 'bid', 'RT', 'sacc_count', 'avg_sacc_duration',
                            'avg_sacc_X_diff', 'avg_sacc_Y_diff']]

    fixData = fixData[fixData['subjectID'] == subjectID]
    if fixData.empty:
        return 0, 0
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
    predictions = lm.predict(X)
    #score
    score = lm.score(X, y)
    print('Score for subjectId - ', subjectId)
    print(score)

    return subjectId, score


# Load the data (fixation or saccade)
path = os.getcwd()
fixation_data = pd.read_csv(path + "/fixation_data.csv")
saccade_data = pd.read_csv(path + "/saccade_data.csv")
subjects = range(102,126)
stimTypes = ['Face', 'Snack']
allSubjects_scores = []
for stim in stimTypes:
    for subjectId in subjects:
        # stim type - Face or Snack
        subjectId, score = oneSubject_oneStimID_multipleLinear_regression(subjectId, fixation_data, saccade_data, stim)
        subjectScore = [stim, subjectId, score]
        allSubjects_scores.append(subjectScore)

scoresDF = pd.DataFrame(allSubjects_scores, columns=['stimType', 'subjectId', 'score'])
scoresDF.to_csv('scoresDF.csv')