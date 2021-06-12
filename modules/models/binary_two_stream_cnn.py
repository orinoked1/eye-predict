import numpy
from tensorflow.keras.layers import Dense, BatchNormalization
from modules.models import cnn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import pandas as pd
import logging
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
import numpy
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import pandas as pd
import logging
from tensorflow.keras import optimizers
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , Flatten

logger = logging.getLogger(__file__)


# CNN multi input
class BinaryTwoStreamCnn:
    def __init__(self, seed, dataset, saliency, run_name, stim_size, run_number):
        # fix random seed for reproducibility
        numpy.random.seed(seed)
        self.trainMapsX, self.valMapsX, self.testMapsX, \
        self.trainImagesX, self.valImagesX, self.testImagesX, self.trainY, self.valY, self.testY = dataset
        self.model1 = None
        self.model2 = None
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
        mapModel = Sequential()
        mapModel.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        #mapModel.add(Conv2D(32, (3, 3), activation='relu'))
        mapModel.add(MaxPooling2D(pool_size=(2, 2)))
        mapModel.add(Dropout(0.5))
        mapModel.add(Flatten())
        mapModel.add(Dense(8, activation='relu'))
        mapModel.add(Dropout(0.5))
        mapModel.add(Dense(self.num_class, activation='sigmoid'))

        imageModel = Sequential()
        imageModel.add(Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))
        imageModel.add(Conv2D(64, (3, 3), activation='relu'))
        imageModel.add(MaxPooling2D(pool_size=(2, 2)))
        imageModel.add(Dropout(0.5))
        imageModel.add(Flatten())
        imageModel.add(Dense(16, activation='relu'))
        imageModel.add(Dropout(0.5))
        imageModel.add(Dense(self.num_class, activation='sigmoid'))

        for layer in mapModel.layers:
            layer.name = layer.name + str("_map")
        for layer in imageModel.layers:
            layer.name = layer.name + str("_image")

        # create the input to our final set of layers as the *output* of both CNNs
        combinedInput = concatenate([mapModel.output, imageModel.output])

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(4, activation="relu")(combinedInput)
        x = Dense(self.num_class, activation="sigmoid")(x)

        # our final model will accept fixation map on one CNN
        # input and images on the second CNN input, outputting a single value as high or low bid (1/0)
        self.model = Model(inputs=[mapModel.input, imageModel.input], outputs=x)

        optimizer = optimizers.Adam(lr=0.0001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print(self.model.summary())

        return

    def train_model(self):
        # shuffle data
        trainMapsX, trainImagesX, trainY = shuffle(self.trainMapsX, self.trainImagesX, self.trainY, random_state=self.seed)
        valMapsX, valImagesX, valY = shuffle(self.valMapsX, self.valImagesX, self.valY, random_state=self.seed)

        #trainY = to_categorical(trainY)
        #valY = to_categorical(valY)

        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(
            [trainMapsX, trainImagesX], trainY,
            validation_data=([valMapsX, valImagesX], valY),
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