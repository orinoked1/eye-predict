import numpy
from tensorflow.keras.layers import Dense
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

logger = logging.getLogger(__file__)


# CNN multi input
class CnnStackedFramesImageConcat:
    def __init__(self, seed, dataset, saliency, run_name, stim_size, patch_size, run_number, num_patches):
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
        self.num_epochs = 60
        self.patch_size = patch_size
        self.stim_size = stim_size
        self.num_class = 10
        self.num_patches = num_patches
        self.lr = 0.0001
        self.momentum = 0.9
        self.decay_rate = self.lr / self.num_epochs

    def define_model(self):
        # create the two CNN models
        cnn_map = cnn.cnn_for_image_concat(self.patch_size, self.patch_size, self.channel * self.num_patches)
        cnn_image = cnn.image_vggNet(self.stim_size[0], self.stim_size[1], self.channel)

        # create the input to our final set of layers as the *output* of both CNNs
        combinedInput = concatenate([cnn_map.output, cnn_image.output])

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(16, activation="relu")(combinedInput)
        x = Dense(self.num_class, activation="softmax")(x)

        # our final model will accept fixation map on one CNN
        # input and images on the second CNN input, outputting a single value as high or low bid (1/0)
        self.model = Model(inputs=[cnn_map.input, cnn_image.input], outputs=x)

        optimizer = optimizers.SGD(lr=self.lr, momentum=self.momentum)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

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
        plt.plot(self.history.history['sparse_categorical_accuracy'])
        plt.plot(self.history.history['val_sparse_categorical_accuracy'])
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

        # shuffle data
        testMapsX, testImagesX, testY = shuffle(self.testMapsX, self.testImagesX, self.testY, random_state=self.seed)
        # shuffle data
        # testPatchesX, testY = shuffle(testPatchesX, testY, random_state=seed)

        # model evaluate
        results = self.model.evaluate([testMapsX, testImagesX], testY, batch_size=None)
        print('test loss, test acc:', results)
        results_df = pd.DataFrame(results, columns=[self.run_name + ", loss, acc"])
        results_df.to_csv(currpath + "/etp_data/processed/" + str(self.run_number) + "_results.csv", index=False)