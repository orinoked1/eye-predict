import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.utils import shuffle
import os
#matplotlib inline

from modules.data.datasets import DatasetBuilder

datasetbuilder = DatasetBuilder()
face_data = datasetbuilder.summery_statistics_features('Face')
Snack_data = datasetbuilder.summery_statistics_features('Snack')

#data_by_stim = data[data['stimName'] == '2_Apropo.jpg']

x = face_data.avg_l2_dist
x = np.array(x).reshape(-1, 1)
y = np.array(face_data.bid)

#Scale / Standardize the features
#sc = StandardScaler()
#x = sc.fit_transform(x)

fig = plt.figure(0)
plt.clf()
plt.hist(x)
plt.xlim(0, 400)
plt.ylim(0, 3500)
currpath = os.getcwd()
fig.savefig(currpath + "/face_dist_hist.pdf", bbox_inches='tight')

fig = plt.figure(1)
plt.clf()
plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.xlabel('dist')
plt.ylabel('rank')
plt.xlim(0, 400)
currpath = os.getcwd()
fig.savefig(currpath + "/snack_fig.pdf", bbox_inches='tight')

linearModel = linear_model.LinearRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
print(linearModel)

fig = plt.figure(2)
plt.clf()
plt.scatter(x, y, color='black')
plt.plot(x, linearModel.predict(x), color='blue', linewidth=3)
plt.legend(loc='best')
currpath
fig.savefig(currpath + "/face_fit.pdf", bbox_inches='tight')
