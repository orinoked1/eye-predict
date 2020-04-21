import numpy
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import pandas as pd
import logging
from keras import optimizers
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Dense, Conv2D , Flatten

logger = logging.getLogger(__file__)


# CNN multi input
class BinarySimpleCnn:
    def __init__(self, seed, dataset, saliency, run_name, stim_size, run_number):
        # fix random seed for reproducibility
        numpy.random.seed(seed)
        self.trainMapsX, self.valMapsX, self.testMapsX, \
        self.trainImagesX, self.valImagesX, self.testImagesX, self.trainY, self.valY, self.testY = dataset
        self.model = None
        self.history = None
        if saliency:
            self.channel = 1
        else:
            self.channel = 3
        self.run_name = run_name
        self.run_number = run_number
        self.seed = seed
        self.batch_size = 32
        self.num_epochs = 100
        self.stimSize = stim_size
        self.num_class = 1

    def define_model(self):
        input_shape = (self.stimSize[1], self.stimSize[0], self.channel)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation='sigmoid'))

        self.model = model

        optimizer = optimizers.SGD(lr=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print(self.model.summary())

        return

    def train_model(self):
        # shuffle data
        trainMapsX, trainImagesX, trainY = shuffle(self.trainMapsX, self.trainImagesX, self.trainY, random_state=self.seed)
        valMapsX, valImagesX, valY = shuffle(self.valMapsX, self.valImagesX, self.valY, random_state=self.seed)


        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(
            trainMapsX, trainY,
            validation_data=(valMapsX, valY),
            epochs=self.num_epochs, batch_size=self.batch_size)


    def metrices(self, currpath):
        # plot metrics
        # summarize history for accuracy
        fig = plt.figure(2)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.savefig(currpath + "/etp_data/processed/figs/" + str(self.run_number) + "_train_val_acc.pdf", bbox_inches='tight')
        plt.show()
        # summarize history for loss
        fig = plt.figure(3)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.savefig(currpath + "/etp_data/processed/figs/" + str(self.run_number) + "_train_val_loss.pdf", bbox_inches='tight')
        plt.show()

        train_acc = self.history.history['accuracy']
        dev_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        dev_loss = self.history.history['val_loss']

        results_df = pd.DataFrame(train_acc, columns=[['train_acc']])
        results_df['dev_acc'] = dev_acc
        results_df['train_loss'] = train_loss
        results_df['dev_loss'] = dev_loss
        results_df.to_csv(currpath + "/etp_data/processed/" + str(self.run_number) + "_results.csv", index=False)
        print('done')

        # shuffle data
        #testMapsX, testImagesX, testY = shuffle(self.testMapsX, self.testImagesX, self.testY, random_state=self.seed)
        # shuffle data
        # testPatchesX, testY = shuffle(testPatchesX, testY, random_state=seed)

        # model evaluate
        #results = self.model.evaluate(testImagesX, testY, batch_size=None)
        #print('test loss, test acc:', results)
        #results_df = pd.DataFrame(results, columns=[self.run_name + ", loss, acc"])
        #results_df.to_csv(currpath + "/etp_data/processed/" + str(self.run_number) + "_results.csv", index=False)