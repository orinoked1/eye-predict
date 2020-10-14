import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import random

datasetbuilder = DatasetBuilder()
face_data = datasetbuilder.summery_statistics_features('Face')
Snack_data = datasetbuilder.summery_statistics_features('Snack')

data_by_stim = face_data

rScores = []
svrScores = []
#for stim in face_data.stimName.unique():
#    data_by_stim = face_data[face_data['stimName'] == stim]

#for subject in face_data.subjectID.unique():
#    data_by_stim = face_data[face_data['subjectID'] == subject]

x = data_by_stim[['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist']]
x = np.array(x)#.reshape(-1, 1)
y = np.array(data_by_stim.bid)

#Scale / Standardize the features
sc = StandardScaler()
x = sc.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

linearModel = linear_model.LinearRegression()
fit = linearModel.fit(X_train, y_train)
score = linearModel.score(X_test, y_test)
pre = linearModel.predict(X_test)
rScores.append(score)

# linear SVR
regr = SVR(kernel='linear', gamma='auto')
svmFit = regr.fit(X_train, y_train)
scoreLinearSvr = regr.score(X_test, y_test)
svrScores.append(scoreLinearSvr)

reg_mean_scores = np.sum(rScores)/len(rScores)
print('Reg mean score: ')
print(reg_mean_scores)
svr_mean_scores = np.sum(svrScores)/len(svrScores)
print('svr mean score: ')
print(svr_mean_scores)

fig = plt.figure(2)
plt.clf()
plt.scatter(x, y, color='black')
plt.plot(x, linearModel.predict(x), color='blue', linewidth=3)
plt.legend(loc='best')
plt.xlabel('Fixations Count')
plt.ylabel('Rank')
plt.title('Linear Regression Fitted Line')
currpath = os.getcwd()
fig.savefig(currpath + "/" + stim + "_linear_fit.pdf", bbox_inches='tight')
