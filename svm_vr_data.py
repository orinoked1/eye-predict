from keras.preprocessing import sequence
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import os
import ds_readers as ds

np.random.seed(31415)


path = os.getcwd()
df = pd.read_csv(path + "/vr/et_for_timeseries_subs_.csv")

# prepering the data
df.dropna(inplace=True)
df = df[df.len > 120]  # 75%
df = df.reset_index()
df['binary_Label'] = pd.qcut(df.Label, 2, labels=[0, 1])

try:
    df = pd.read_pickle("vr/vr_scanpath_df.pkl")
except:
    # create scanpath column
    df['scanpath'] = None
    df['scanpath'] = df['scanpath'].astype(object)
    for sampleId in df.SampleID.unique():
        print('Building scanpath dataset to sampleId: ' + sampleId)
        scanpath = ds.vr_data_to_scanpath(df, sampleId, 4)
        listIndex = df.index[df['SampleID'] == sampleId].unique()
        for index in listIndex:
            df.at[index, 'scanpath'] = scanpath

    df.to_pickle("vr/vr_scanpath_df.pkl")

try:
    df = pd.read_pickle("vr/vr_scanpath__xy_only_df.pkl")
except:
    # create scanpath column
    df['scanpath_xy_only'] = None
    df['scanpath_xy_only'] = df['scanpath_xy_only'].astype(object)
    for sampleId in df.SampleID.unique():
        print('Building scanpath xy only dataset to sampleId: ' + sampleId)
        scanpath = ds.vr_xy_only_data_to_scanpath(df, sampleId, 4)
        listIndex = df.index[df['SampleID'] == sampleId].unique()
        for index in listIndex:
            df.at[index, 'scanpath_xy_only'] = scanpath

    df.to_pickle("vr/vr_scanpath__xy_only_df.pkl")

df = df.drop_duplicates('SampleID')
df.reset_index(inplace=True)

X = df.scanpath_xy_only
y = df.binary_Label
X = np.asanyarray(X)
y = np.asanyarray(y)
# Set the vector length - milliseconds size window.
max_vector_length = 120
cut_scanpath = []
for i in X:
    cut_scanpath.append(i[:120])
X = np.asanyarray(cut_scanpath)
#X = sequence.pad_sequences(X, maxlen=max_vector_length)

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

svm_scores_df = pd.DataFrame(scores, columns=['scores'])
svm_scores_df.to_csv("vr_xy_only_svm_scores_df.csv")

#scores = np.asanyarray(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


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

svm_parmotation_scores_df = pd.DataFrame(scores, columns=['scores'])
svm_parmotation_scores_df.to_csv("svm_faces_parmotation_scores_df.csv")