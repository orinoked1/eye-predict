import numpy
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import pandas as pd
import logging
from keras import optimizers
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D
from keras import regularizers

logger = logging.getLogger(__file__)


# CNN multi input
class BinarySimpleCnn:
    def __init__(self, seed, run_name, stim_size, color_split):
        # fix random seed for reproducibility
        self.seed = seed
        numpy.random.seed(self.seed)
        self.model = None
        self.history = None
        self.colorpathSplit = color_split
        self.channel = 3
        self.run_name = run_name
        self.batch_size = 128
        self.num_epochs = 100
        self.stimSize = stim_size
        self.num_class = 1
        self.LR = 0.001
        self.optimizer = None
        self.loss_function = None
        self.stimType = 'Face'
        self.input_type = 'fixation-map'
        self.datapath = "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/"


    def define_model(self):
        input_shape = (self.stimSize[1], self.stimSize[0], self.channel)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(self.num_class, activation='sigmoid'))

        self.model = model

        self.optimizer = optimizers.SGD(lr=0.0001)
        self.loss_function = 'binary_crossentropy'
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
        results = self.model.evaluate(testX, testY, batch_size=self.batch_size)
        print('test loss, test acc:', results)


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

        train_acc = self.history.history['accuracy']
        dev_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        dev_loss = self.history.history['val_loss']

        results_df = pd.DataFrame(train_acc, columns=[['train_acc']])
        results_df['dev_acc'] = dev_acc
        results_df['train_loss'] = train_loss
        results_df['dev_loss'] = dev_loss
        results_df.to_csv(self.datapath + str(self.run_name) + "_results.csv", index=False)

        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)

        resultsDF = pd.read_csv(self.datapath + "binary_cnn_results.csv")
        model_to_append = [str(self.run_name), self.batch_size, self.num_epochs, self.LR,
                           self.seed, self.stimType, self.optimizer, self.loss_function, self.input_type,
                           self.colorpathSplit, short_model_summary,
                           max(train_acc)*100, max(dev_acc)*100, min(train_loss),
                           min(dev_loss)]

        a_series = pd.Series(model_to_append, index=resultsDF.columns)
        resultsDF = resultsDF.append(a_series, ignore_index=True)
        resultsDF.to_csv(self.datapath + "binary_cnn_results.csv", index=False)


        print('done')