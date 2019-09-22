from sklearn.utils import shuffle
from keras.preprocessing import sequence
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesClassifier as tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import os
import ds_readers as ds

np.random.seed(31415)


path = os.getcwd()

try:
    df = pd.read_pickle("etp_scanpath_df.pkl")
except:
    raw_data_df_101_117 = pd.read_csv(path + '/output_data_both_eyes_101_117.csv')
    raw_data_df_118_125 = pd.read_csv(path + '/output_data_both_eyes_118_125.csv')
    allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])

    scanpath_dataset = ds.get_scanpath_dataset(allSubjectsData, ([1080, 1920]))
    scanpath_df = pd.DataFrame(scanpath_dataset)
    scanpath_df.columns = ['subjectID', 'stimName', 'stimType', 'sampleId', 'scanpath', 'bid']
    scanpath_df.to_pickle("etp_scanpath_df.pkl")


try:
    df = pd.read_pickle("df.pkl")
except:
    df['scanpath_len'] = 0
    for i in range(df.scanpath.size):
        print(i)
        df['scanpath_len'][i] = len(df.scanpath[i])
    df.to_pickle("df.pkl")

# prepering the data
df = df[df.scanpath_len > 2300]  # > 75%
df = df.reset_index()
df = df[df['stimType'] == "Snack"]

df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])

X = df.scanpath
y = df.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)
# truncate and pad input sequences
max_review_length = 500
X = sequence.pad_sequences(X, maxlen=max_review_length)

dataset_size = len(X)
X = X.reshape(dataset_size,-1)

#cross validation correct lables
scores = []
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # clf = tree()
    clf = svm.SVC(kernel="rbf", gamma=0.0000001)
    clf.fit(X_train, y_train)
    print("Train score: ", clf.score(X_train, y_train))
    print("Test score: ", clf.score(X_test, y_test))
    scores.append(clf.score(X_test, y_test))

scores = np.asanyarray(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# parmotation runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

scores = []
for i in range(1000):
    print("Permotation number: %d" % i)
    y_train = np.random.permutation(y_train)
    clf = svm.SVC(kernel="rbf", gamma=0.0000001)
    clf.fit(X_train,y_train)
    print("Train score: ", clf.score(X_train,y_train))
    print("Test score: ", clf.score(X_test,y_test))
    scores.append(clf.score(X_test,y_test))
scores = np.asanyarray(scores)