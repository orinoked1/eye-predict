import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import os

from modules.data.datasets import DatasetBuilder

datasetbuilder = DatasetBuilder()

datasetbuilder.participents_s_data()

face_data = datasetbuilder.summery_statistics_features('Face')
Snack_data = datasetbuilder.summery_statistics_features('Snack')

#data_by_stim = data[data['stimName'] == '2_Apropo.jpg']

x = Snack_data.avg_l2_dist
x = np.array(x).reshape(-1, 1)
y = np.array(Snack_data.bid)

#Scale / Standardize the features
#sc = StandardScaler()
#x = sc.fit_transform(x)

fig = plt.figure(0)
plt.clf()
plt.hist(x)
plt.xlim(0, 400)
plt.ylim(0, 3500)
plt.xlabel('L2 Average Distances')
plt.ylabel('Count')
plt.title('L2 average distances for Face stimuli')
currpath = os.getcwd()
fig.savefig(currpath + "/snack_dist_hist.pdf", bbox_inches='tight')


linearModel = linear_model.LogisticRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
print(linearModel)

fig = plt.figure(2)
plt.clf()
plt.scatter(x, y, color='black')
plt.plot(x, linearModel.predict(x), color='blue', linewidth=3)
plt.legend(loc='best')
plt.xlabel('L2 Average Distance')
plt.ylabel('Rank')
plt.title('Logistic regression fited line')
currpath
fig.savefig(currpath + "/face_logistic_fit.pdf", bbox_inches='tight')
