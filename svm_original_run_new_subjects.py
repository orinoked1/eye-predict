from keras.preprocessing import sequence
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle

np.random.seed(31415)


path = os.getcwd()
"""
try:
    df = pd.read_pickle(path + "/etp_data/processed/scanpath_df_126_128.pkl") #df = pd.read_pickle("processed_data/etp_scanpath_df.pkl")
except:
    #raw_data_df_101_117 = pd.read_csv(path + '/output_data_both_eyes_101_117.csv')
    #raw_data_df_118_125 = pd.read_csv(path + '/output_data_both_eyes_118_125.csv')
    #allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])

    df = pd.read_pickle(path + "/etp_data/processed/tidy_data_126_128.pkl")
    scanpath_dataset = ds.get_scanpath_dataset(df, ([1080, 1920]))
    scanpath_df = pd.DataFrame(scanpath_dataset)
    scanpath_df.columns = ['subjectID', 'stimName', 'stimType', 'sampleId', 'scanpath', 'bid']
    scanpath_df.to_pickle(path + "/etp_data/processed/tidy_data_126_128.pkl")
"""

try:
    new_data = pd.read_pickle(path + "/etp_data/processed/scanpath_df_126_128_len.pkl") #df = pd.read_pickle("processed_data/df.pkl")
    old_data = pd.read_pickle("processed_data/df.pkl")
except:
    df['scanpath_len'] = 0
    for i in range(df.scanpath.size):
        print(i)
        df['scanpath_len'][i] = len(df.scanpath[i])
    df.to_pickle(path + "/etp_data/processed/scanpath_df_126_128_len.pkl") #df.to_pickle("df.pkl")

udf = pd.concat([new_data, old_data])

# prepering the data
udf = udf[udf.scanpath_len > 2300]  # > 75%
udf = udf.reset_index()
# Set the relevant stim
udf = udf[udf['stimType'] == "Snack"]
udf['binary_bid'] = pd.qcut(udf.bid, 2, labels=[0, 1])

X = udf.scanpath
y = udf.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)

# Set the vector length - milliseconds size window.
max_vector_length = 1500
X = sequence.pad_sequences(X, maxlen=max_vector_length)

dataset_size = len(X)
X = X.reshape(dataset_size, -1)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#clf = svm.SVC(kernel="rbf", gamma=0.0000001)
#clf.fit(X_train, y_train)
#print("Train score: ", clf.score(X_train, y_train))
#print("Test score: ", clf.score(X_test, y_test))

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

print("ACC: ", (sum(scores) / len(scores))*100)
svm_scores_df = pd.DataFrame(scores, columns=['scores'])
svm_scores_df.to_csv("svm_faces_scores_df.csv")

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
