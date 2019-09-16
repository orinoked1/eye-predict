import os
import ds_readers as ds
import pickle
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras.layers import CuDNNLSTM
import tensorflow as tf

from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


path = os.getcwd()

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

# prepering the data
df = df[df.scanpath_len > 2300]  # > 75%
df = df.reset_index()
df = df[df['stimType'] == "Snack"]
# Normalize total_bedrooms column
x_array = np.array(df['bid'])
normalized_x = preprocessing.normalize([x_array])
df['normalized_bid'] = normalized_x.reshape(2472, )
# Normalize total_bedrooms column
x_array = np.array(df['bid'])
normalized_x = preprocessing.normalize([x_array])
df['normalized_bid'] = normalized_x.reshape(2472, )
df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])
X = df.scanpath
y = df.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)
# truncate and pad input sequences
max_review_length = 500
X = sequence.pad_sequences(X, maxlen=max_review_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)



def original_run():
    # No permutations
    # create the model
    model = Sequential()
    if use_gpu:
        model.add(CuDNNLSTM(100, input_shape=(max_review_length, 2)))
    else:
        model.add(LSTM(100, input_shape=(max_review_length, 2)))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, shuffle=False)

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (score[1] * 100))

def permutations_run():
    permotation_scores = []
    for i in range(1000):
        print("Permotation number: %d" % i)
        np.random.permutation(y_train)
        # create the model
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
    scores_df.to_csv("scores_df")

for i in range(4):
    original_run()
print("done")

