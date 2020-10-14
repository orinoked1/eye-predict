import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import random

datasetbuilder = DatasetBuilder()
face_data = datasetbuilder.summery_statistics_features('Face')
Snack_data = datasetbuilder.summery_statistics_features('Snack')

stimType = 'Snack'
if stimType == 'Face':
    data = face_data
elif stimType == 'Snack':
    data = Snack_data



#classification SVM
#data['five_bin_bid'] = pd.qcut(data.bid, 5, labels=[0, 1, 2, 3, 4])
scores = []
for stim in data.stimName.unique():
    data_by_stim = data[data['stimName'] == stim]
#for subject in data.subjectID.unique():
    #data_by_stim = data[data['subjectID'] == subject]

    data_by_stim['five_bin_bid'] = pd.qcut(data_by_stim.bid, 5, labels=[0, 1, 2, 3, 4])

    x = data_by_stim[['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist']]
    x = np.array(x)#.reshape(-1, 1)
    y = np.array(data_by_stim.five_bin_bid)

    #Scale / Standardize the features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    regr = SVC(kernel='rbf', gamma='auto')
    svmFit = regr.fit(X_train, y_train)
    score = regr.score(X_test, y_test)
    scores.append(score)

mean_scores = np.mean(scores)
print(mean_scores)
"""
Permutations_score = []
for i in range(1000):
    print(i)
    #classification SVM
    data['five_bin_bid'] = pd.qcut(data.bid, 5, labels=[0, 1, 2, 3, 4])
    scores = []
    for stim in data.stimName.unique():
        data_by_stim = data[data['stimName'] == stim]

        x = data_by_stim[['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist']]
        x = np.array(x)#.reshape(-1, 1)
        y = np.array(data_by_stim.five_bin_bid)
        np.random.shuffle(y)

        #Scale / Standardize the features
        sc = StandardScaler()
        x = sc.fit_transform(x)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        regr = SVC(kernel='rbf', gamma='auto')
        svmFit = regr.fit(X_train, y_train)
        score = regr.score(X_test, y_test)
        scores.append(score)

    mean_scores = np.sum(scores)/len(scores)
    Permutations_score.append(mean_scores)

print('Max permutation value: ')
print(max(Permutations_score))
print('Min permutation value: ')
print(min(Permutations_score))

fig = plt.figure(0)
plt.clf()
plt.hist(Permutations_score)
plt.title('Permutation Scores')
currpath = os.getcwd()
fig.savefig(currpath + "/snack_svm_permutation_hist.pdf", bbox_inches='tight')
"""

"""
fig = plt.figure(2)
plt.clf()
plt.scatter(x, y, color='black')
plt.plot(x, svmFit.predict(x), color='blue', linewidth=3)
plt.legend(loc='best')
plt.xlabel('Fixations Count')
plt.ylabel('Rank')
plt.title('Linear regression fited line')
currpath = os.getcwd()
fig.savefig(currpath + "/" + stim + "_linear_fit.pdf", bbox_inches='tight')
"""
