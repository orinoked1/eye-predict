import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import random
import seaborn as sns
import os

stimType = 'Face'

datasetbuilder = DatasetBuilder()
df = datasetbuilder.summery_statistics_features(stimType)

# save df to csv
df_to_save = df[['sampleId', 'subjectID', 'stimName', 'avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist', 'bid']]
currpath = os.getcwd()
df_to_save.to_csv(currpath + "/face_ss_features.csv")


# x y split
x = df[['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist']]
x = np.array(x)
y = np.array(df.bid)

#Scale / Standardize the features
sc = StandardScaler()
x = sc.fit_transform(x)

# linear regression
linearModel = linear_model.LinearRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
predictions = linearModel.predict(x)
rmse = np.sqrt(np.square(np.subtract(y, predictions)).mean())

# Ordinary Least Squares
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())





# calculate features correlations and make a seaborn heatmap

data = df[['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist']]

corr = data.corr()
fig = plt.figure(2)
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

currpath = os.getcwd()
fig.savefig(currpath + "/face_visual_features_heat_map.pdf", bbox_inches='tight')




print('x')