import numpy
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import pandas as pd
import logging
from keras import optimizers
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Dense, Conv2D , Flatten
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from keras import backend as K

logger = logging.getLogger(__file__)


# CNN multi input
class BinaryNN:
    def __init__(self, seed, run_name, stim_size, scanpath_lan, stimType, bin_count):
        # fix random seed for reproducibility
        self.seed = seed
        numpy.random.seed(self.seed)
        self.model = None
        self.history = None
        self.scanpath_lan = scanpath_lan
        self.run_name = run_name
        self.batch_size = 128
        self.num_epochs = 400
        self.stimSize = stim_size
        self.num_class = 1
        self.LR = 0.001
        self.optimizer = None
        self.loss_function = None
        self.stimType = stimType
        self.input_type = 'scanpath'
        self.datapath = "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/"
        self.corr = None
        self.bin_count = bin_count

    def root_mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def define_model(self):
        input_shape = (self.scanpath_lan, 3)
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        #model.add(Dense(32, kernel_initializer=initializers.RandomNormal(), activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', activity_regularizer=regularizers.l1(1e-5)))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', activity_regularizer=regularizers.l1(1e-5)))
        #model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='linear'))

        self.model = model
        self.optimizer = optimizers.SGD(lr=self.LR)
        self.loss_function = self.root_mean_squared_error
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['accuracy'])

        print(self.model.summary())

        return

    def train_model(self, trainX, trainY, valX, valY):
        # shuffle data
        trainX, trainY = shuffle(trainX, trainY, random_state=self.seed)
        valX, valY = shuffle(valX, valY, random_state=self.seed)

        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(
            trainX, trainY,
            validation_data=(valX, valY),
            epochs=self.num_epochs, batch_size=self.batch_size)

    def test_model(self, testX, testY):
        # shuffle data
        testX, testY = shuffle(testX, testY, random_state=self.seed)

        print('[INFO] Evaluate on test data')
        y_pred = self.model.predict(testX)
        y_pred = y_pred.flatten()
        self.corr = numpy.corrcoef(y_pred, testY)
        print(self.corr)


    def metrices(self):
        # plot metrics
        # summarize history for accuracy
        fig = plt.figure(2)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.savefig(self.datapath + "figs/" + str(self.run_name) + "_train_val_acc.pdf", bbox_inches='tight')
        plt.show()
        # summarize history for loss
        fig = plt.figure(3)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.savefig(self.datapath + "figs/" + str(self.run_name) + "_train_val_loss.pdf", bbox_inches='tight')
        plt.show()

        dev_corr = self.corr[0][1]
        train_loss = self.history.history['loss']
        dev_loss = self.history.history['val_loss']

        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)

        resultsDF = pd.read_csv(self.datapath + "feedForwardNet_results.csv")
        model_to_append = [str(self.run_name), self.batch_size, self.num_epochs, self.LR,
                           self.seed, self.stimType, self.optimizer, self.loss_function, self.input_type,
                           self.scanpath_lan, self.bin_count, short_model_summary, dev_corr, min(train_loss),
                           min(dev_loss)]

        a_series = pd.Series(model_to_append, index=resultsDF.columns)
        resultsDF = resultsDF.append(a_series, ignore_index=True)
        resultsDF.to_csv(self.datapath + "feedForwardNet_results.csv", index=False)


        print('done')