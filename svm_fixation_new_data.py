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

try:
    df = pd.read_csv(path + "/etp_data/processed/126_138_one_eye_data.csv")
    df_old = pd.read_pickle(path + "/processed_data/fixation_array_dataset.pkl")
except:
    raw_data_df_101_117 = pd.read_csv(path + '/output_data_both_eyes_101_117.csv')
    raw_data_df_118_125 = pd.read_csv(path + '/output_data_both_eyes_118_125.csv')
    allSubjectsData = pd.concat([raw_data_df_101_117, raw_data_df_118_125])

    scanpath_dataset = ds.get_scanpath_dataset(allSubjectsData, ([1080, 1920]))
    scanpath_df = pd.DataFrame(scanpath_dataset)
    scanpath_df.columns = ['subjectID', 'stimName', 'stimType', 'sampleId', 'scanpath', 'bid']
    scanpath_df.to_pickle("etp_scanpath_df.pkl")

"""
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
"""
try:
    df = pd.read_pickle("processed_data/fixation_array_dataset.pkl")
except:
    #get only right eye data
    df = df[df['eye'] == 'R']
    df.reset_index(inplace=True)
    #create fixation array
    flag = 1
    for sampleId in df.sampleId.unique():
        scanpath = ds.data_to_scanpath(df, sampleId, 4)
        print('here')
        listIndex = df.index[df['sampleId'] == sampleId].unique()

        if flag:
            df['fixation_array'] = None
            df['fixation_array'] = df['fixation_array'].astype(object)
        flag = 0
        for index in listIndex:
            print(listIndex)
            df.at[index, 'fixation_array'] = scanpath

    #add binary bid column
    df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])

    df.to_pickle("fixation_array_dataset.pkl")

df.dropna(inplace=True)
df.reset_index(inplace=True)
df = df.drop_duplicates('sampleId')

# Set the relevant stim
df = df[df['stimType'] == "Snack"]

X = df.fixation_array
y = df.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)
# Set the vector length - milliseconds size window.
max_vector_length = 17
X = sequence.pad_sequences(X, maxlen=max_vector_length)

dataset_size = len(X)
X = X.reshape(dataset_size, -1)

#cross validation correct lables
scores = []
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # clf = tree()
    clf = svm.SVC(kernel="rbf", gamma=0.00001)
    clf.fit(X_train, y_train)
    print("Train score: ", clf.score(X_train, y_train))
    print("Test score: ", clf.score(X_test, y_test))
    scores.append(clf.score(X_test, y_test))

print("ACC: ", (sum(scores) / len(scores))*100)
svm_scores_df = pd.DataFrame(scores, columns=['scores'])
svm_scores_df.to_csv("fixations_svm_snack_scores_df.csv")

#scores = np.asanyarray(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# parmotation runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

y_trainlist = []
scores = []
for i in range(1000):
    print("Permotation number: %d" % i)
    y_train = np.random.permutation(y_train)
    y_trainlist.append(y_train)
    clf = svm.SVC(kernel="rbf", gamma=0.00001)
    clf.fit(X_train,y_train)
    print("Train score: ", clf.score(X_train,y_train))
    print("Test score: ", clf.score(X_test,y_test))
    scores.append(clf.score(X_test,y_test))

svm_parmotation_scores_df = pd.DataFrame(scores, columns=['scores'])
svm_parmotation_scores_df.to_csv("fixations_svm_snack_parmotation_scores_df.csv")