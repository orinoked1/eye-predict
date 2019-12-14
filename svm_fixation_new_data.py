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

"""
try:
    df = pd.read_csv(path + "/etp_data/processed/126_138_one_eye_data.csv")
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
"""
try:
    old_data = pd.read_pickle(path + "/processed_data/fixation_array_dataset.pkl")
    new_data = pd.read_pickle(path+ "/etp_data/processed/scanpath_df_fixation_126_128.pkl")
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

old_data.dropna(inplace=True)
old_data = old_data.drop_duplicates('sampleId')
old_data.reset_index(inplace=True)
old_data = old_data[['stimName', 'stimType', 'sampleId', 'fixation_array', 'bid']]
old_data.rename(columns={"fixation_array": "scanpath"}, inplace=True)

df = old_data #pd.concat([old_data, new_data])
#add binary bid column
df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])

# Set the relevant stim
df = df[df['stimType'] == "Face"]

X = df.scanpath
y = df.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)
# Set the vector length - milliseconds size window.
max_vector_length = 17
X = sequence.pad_sequences(X, maxlen=max_vector_length)

dataset_size = len(X)
X = X.reshape(dataset_size, -1)

#add binary bid column
new_data['binary_bid'] = pd.qcut(new_data.bid, 2, labels=[0, 1])

# Set the relevant stim
new_data = new_data[new_data['stimType'] == "Snack"]

test_X = new_data.scanpath
test_y = new_data.binary_bid
test_X = np.asanyarray(test_X)
test_y = np.asanyarray(test_y)
# Set the vector length - milliseconds size window.
max_vector_length = 17
test_X = sequence.pad_sequences(test_X, maxlen=max_vector_length)

dataset_size = len(test_X)
test_X = test_X.reshape(dataset_size, -1)

X_traintt, test_X, y_traintt, test_y = train_test_split(
    test_X, test_y, test_size=0.33, random_state=42)

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
    print("Test score: ", clf.score(test_X, test_y))
    scores.append(clf.score(test_X, test_y))

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