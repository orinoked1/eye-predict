import os
import ds_readers as ds
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn import preprocessing
import tensorflow as tf

from numpy.random import seed
seed(11)
from tensorflow import set_random_seed
set_random_seed(2)

path = os.getcwd()
"""
try:
    df = pd.read_pickle(path + "/etp_scanpath_df.pkl")
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

use_gpu = tf.test.is_gpu_available()
print("use GPU?:",use_gpu )

try:
    df = pd.read_pickle("df_image_array.pkl")
except:
    # prepering the data
    df = df[df.scanpath_len > 2300]  # > 75%
    df = df.reset_index()
    #get image arrays
    flag = 1
    for stim in df.stimName.unique():
        print('building image array for stim: ' + stim)
        newpath = path + "/etp_data/Stim_0/" + stim
        im = cv2.imread(newpath, 0)
        listIndex = df.index[df['stimName'] == stim].unique()

        if flag:
            df['img_array'] = None
            df['img_array'] = df['img_array'].astype(object)
        flag = 0
        for index in listIndex:
            df.at[index, 'img_array'] = im

    df.to_pickle("df_image_array.pkl")


#Face stim data preperation
df = df[df['stimType'] == "Face"]
df.img_array = df.img_array.apply(lambda x: np.asanyarray(cv2.resize(x, (400, 500))))
# truncate and pad input sequences
max_scanpath_length = 2800
df['padded_scanpath'] = sequence.pad_sequences(df.scanpath, maxlen=max_scanpath_length, padding='post', value=0).tolist()
df.padded_scanpath = df.padded_scanpath.apply(lambda x: x[::7])
df.padded_scanpath = df.padded_scanpath.apply(lambda x: np.asanyarray(x))
data_concatenate = []
for val1, val2 in zip(df.img_array, df.padded_scanpath):
    val1 = val1.T
    data_concatenate.append(np.concatenate((val2, val1), axis=1))

df['data_concatenate'] = data_concatenate

df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])


X = np.asanyarray(df.data_concatenate.tolist())
y = np.asanyarray(df.binary_bid.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)
X_train_shape = X_train.shape
X_val_shape = X_val.shape
#y_train_shape = y_train.shape
#y_val_shape = y_val.shape
X_train = X_train.reshape(X_train_shape[0],X_train_shape[1],X_train_shape[2], 1)
X_val = X_val.reshape(X_val_shape[0],X_val_shape[1],X_val_shape[2], 1)
#y_train = y_train.reshape(y_train_shape[0], 1)
#y_val = y_val.reshape(y_val_shape[0], 1)

zzz = X_train.shape[1:]
print(max_scanpath_length)

def original_run():
    # No permutations
    intialization_scores = []
    for i in range(1):
        print("Intialization number: %d" % i)
        # create the model
        model = Sequential()
        with tf.device('cpu'):
            model.add(Dense(12, input_shape=X_train.shape[1:], activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # print(model.summary())
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, shuffle=False)

            # Final evaluation of the model
            score = model.evaluate(X_test, y_test, verbose=0)
            intialization_scores.append(score[1] * 100)
            print("Accuracy: %.2f%%" % (score[1] * 100))
    intialization_scores_df = pd.DataFrame(intialization_scores, columns = ['scores'])
    intialization_scores_df.to_csv("faceStim_intialization_scores_df.csv")

def permutations_run():
    permotation_scores = []
    for i in range(1000):
        print("Permotation number: %d" % i)
        np.random.permutation(y_train)
        # create the model
        with tf.device('gpu'):
            model = Sequential()
            model.add(LSTM(100, input_shape=(max_review_length, 2)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # print(model.summary())
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, shuffle=False)

            # Final evaluation of the model
            scores = model.evaluate(X_test, y_test, verbose=0)
            permotation_scores.append(scores[1] * 100)
            print("Accuracy: %.2f%%" % (scores[1] * 100))

    scores_df = pd.DataFrame(permotation_scores, columns=['scores'])
    scores_df.to_csv("faceStim_permotation_scores_df.csv")


original_run()
permutations_run()

print("done")

