import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVR, SVC
from sklearn.utils import shuffle
#matplotlib inline

from modules.data.datasets import DatasetBuilder

datasetbuilder = DatasetBuilder()
face_data = datasetbuilder.summery_statistics_features('Face')
Snack_data = datasetbuilder.summery_statistics_features('Snack')
all_data = pd.concat([face_data, Snack_data])
all_data = shuffle(all_data)

#data_by_stim = data[data['stimName'] == '2_Apropo.jpg']

x = face_data.avg_l2_dist
x = np.array(x).reshape(-1, 1)
#face_data['binary_bid'] = pd.qcut(Snack_data.bid, 2, labels=[0, 1])
y = np.array(face_data.bid)
#y = y - 1
#Scale / Standardize the features
#sc = StandardScaler()
#x = sc.fit_transform(x)

fig = plt.figure(1)
plt.clf()
plt.plot(x, y, 'go', label='True data', alpha=0.5)
#plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
fig.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/torch/linear_regression/" + "fig.pdf", bbox_inches='tight')

linearModel = linear_model.LinearRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
print(linearModel)

fig = plt.figure(2)
plt.scatter(x, y, color='black')
plt.plot(x, linearModel.predict(x), color='blue', linewidth=3)
fig.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/torch/linear_regression/" + "fit.pdf", bbox_inches='tight')

regr = SVR()
svmFit = regr.fit(x, y)
svmScore = regr.score(x, y)

fig = plt.figure(3)
plt.scatter(x, y, color='black')
plt.plot(x, regr.predict(x), color='blue', linewidth=3)
fig.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/torch/linear_regression/" + "svm_fit.pdf", bbox_inches='tight')


data = pd.read_excel('nergy_data.xlsx')
print(data.head(5))

#seperating data
X = data[:,:4]
y = data[:,-1]

#Scale / Standardize the features
sc = StandardScaler()
X = sc.fit_transform(X)

#Defining cost MSE
def cost_function(X, Y, B):
 m = len(Y)
 J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
 return J

#In Batch Gradient Descent Function we take parameters
#X: Feature Matrix
#Y: an array of target values
#B: initial value of theta
#alpha: learning rate
#iterations: max no. of iterations for algorithm
def batch_gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # print(iteration)
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history

#Splitting training and testing sets

m = 7000
f = 2
X_train = X[:m,:f]
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
y_train = y[:m]
X_test = X[m:,:f]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]
y_test = y[m:]

#Initializing the coefficients(β0​, β1​,…, βn) and training the model
B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 2000
newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)

#Evaluating / Testing the model
y_ = pred(X_test,newB)

#now check how good is it predicting using r2 score
def r2(y_,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)
#----------------
r2(y_,y_test)