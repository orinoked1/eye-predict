import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

stimType = 'Face'

datasetbuilder = DatasetBuilder()
fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)

df = datasetbuilder.get_fixations_scanpath_df(fixation_event_data, stimType)
# remove small scanpath samples
df['scanpath_len'] = 0
for i in range(df.scanpath.size):
    df.at[i, 'scanpath_len'] = len(df.scanpath[i])
df = df[df['scanpath_len'] > 5]

stim_avg = df.groupby(['stimName'])['bid'].mean().reset_index()
stim_avg.rename(columns={'bid': 'avg'}, inplace=True)

df = df.merge(stim_avg, how='left', on='stimName')

# x y split
x = np.array(df.avg).reshape(-1, 1)
y = np.array(df.bid).reshape(-1, 1)

# linear regression
linearModel = linear_model.LinearRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
rmse = np.sqrt(np.square(np.subtract(x, y)).mean())

# Ordinary Least Squares
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())

# Mixed linear model
md = smf.mixedlm("bid ~ avg", df, groups=df["subjectID"])
mdf = md.fit()
print(mdf.summary())




print('l')