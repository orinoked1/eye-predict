import os
import ds_readers as ds
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn import preprocessing
import tensorflow as tf
from sklearn.utils import shuffle
import keras.callbacks as cb
from datetime import datetime

from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from numpy.random import seed
seed(11)
from tensorflow import set_random_seed
set_random_seed(2)

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
df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])

df = df[((df['bid'] <= 4)&(df['bid'] >= 2)) | (df['bid'] <= 8)&(df['bid'] >= 6)]
df['2_9'] = np.where(df.bid <= 4, 0, 1)

df.reset_index(inplace=True)
df['2_9'] = df['2_9'].astype(int)
q = df[df['2_9'] == 0].count()
p = df[df['2_9'] == 1].count()

df0 = df[df['2_9'] == 0]
df1 = df[df['2_9'] == 1]

df1 = df1.sample(n=461, random_state=1)
df = pd.concat([df0, df1])
df = shuffle(df, random_state=1)

X = df.scanpath
y = df.binary_bid
X = np.asanyarray(X)
y = np.asanyarray(y)
# truncate and pad input sequences
max_review_length = 1500
X = sequence.pad_sequences(X, maxlen=max_review_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10)

unique, counts = np.unique(y_train, return_counts=True)
y_Train_count = dict(zip(unique, counts))

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = cb.TensorBoard(log_dir=logdir)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def original_run():
    # No permutations
    intialization_scores = []
    for i in range(30):
        print("Intialization number: %d" % i)
        # create the model
        model = Sequential()
        with tf.device('gpu'):
            model.add(LSTM(100, input_shape=(max_review_length, 2)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # print(model.summary())
            history = LossHistory()
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=1, shuffle=False, callbacks=[history, tensorboard_callback])

            print(history.losses)
            # Final evaluation of the model
            score = model.evaluate(X_test, y_test, verbose=0, callbacks=[history])
            intialization_scores.append(score[1] * 100)
            print("Accuracy: %.2f%%" % (score[1] * 100))
    intialization_scores_df = pd.DataFrame(intialization_scores, columns = ['scores'])
    intialization_scores_df.to_csv("face_2_4_6_8_intialization_scores_df.csv")

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
    scores_df.to_csv("permotation_scores_df.csv")


original_run()
#permutations_run()

print("done")

