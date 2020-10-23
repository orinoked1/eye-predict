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

df_to_save = df[['subjectID', 'stimName', 'avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist', 'bid']]
currpath = os.getcwd()
df_to_save.to_csv(currpath + "/face_ss_features_scaled.csv")
#Scale / Standardize the features
df_to_save = np.array(df_to_save)
sc = StandardScaler()
df_to_save = sc.fit_transform(df_to_save)
df_to_save = pd.DataFrame(df_to_save, columns=['avg_fix_duration', 'avg_sacc_duration', 'fix_count', 'sacc_count', 'avg_l2_dist', 'bid'])
df_to_save = pd.merge(df, df_to_save)
df_to_save = df_to_save[['subjectID', 'stimName', 'avg_fix_duration', 'avg_sacc_duration', 'fix_count',
                 'sacc_count', 'avg_l2_dist', 'bid']]
currpath = os.getcwd()
df_to_save.to_csv(currpath + "/face_ss_features_scaled.csv")


#for i, field in zip(range(df_to_save.shape[1]), df_to_save):
#    fig = plt.figure(i)
#    plt.hist(df_to_save[field])
#    currpath = os.getcwd()
#    fig.savefig(currpath + "/" + field + "_feature_dist.pdf", bbox_inches='tight')


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