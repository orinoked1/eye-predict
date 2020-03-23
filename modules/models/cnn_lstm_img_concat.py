import numpy
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from modules.models import cnn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from keras.layers import concatenate
from keras.models import Model
from keras import optimizers


# CNN LSTM image concatenation for sequence classification
class CnnLstmImgConcat:
    def __init__(self, seed, dataset, saliency, patch_size, run_name, stim_size):
        # fix random seed for reproducibility
        numpy.random.seed(seed)
        self.trainPatchesX, self.valPatchesX, self.testPatchesX, \
        self.trainImagesX, self.valImagesX, self.testImagesX, self.trainY, self.valY, self.testY = dataset
        self.model = None
        self.history = None
        if saliency:
            self.channel = 1
        else:
            self.channel = 3
        self.patch_size = patch_size
        self.run_name = run_name
        self.seed = seed
        self.batch_size = 16
        self.num_epochs = 2
        self.stimSize = stim_size
        self.num_class = 10

    def define_model(self):
        # create the two CNN models
        cnn_patchscan = cnn.map_vggNet(self.patch_size, self.patch_size, self.channel)
        cnn_image = cnn.image_vggNet(self.stimSize[0], self.stimSize[1], self.channel)

        # create the input to our final set of layers as the *output* of both CNNs
        combinedInput = concatenate([cnn_patchscan.output, cnn_image.output])

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(32, activation="relu")(combinedInput)
        x = TimeDistributed(Dense(16, activation="relu"), input_shape=(50, 16))(x)
        x = LSTM(16, activation='relu', return_sequences=False)(x)
        x = Dense(self.num_class, activation="softmax")(x)

        # our final model will accept scanpth on one CNN
        # input and images on the second CNN input, outputting a single value as high or low bid (1/0)
        self.model = Model(inputs=[cnn_patchscan.input, cnn_image.input], outputs=x)

        optimizer = optimizers.Adam(lr=0.00001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                           metrics=['sparse_categorical_accuracy'])

        print(self.model.summary())

        return

    def train_model(self):
        # shuffle data
        trainPatchesX, trainImagesX, trainY = shuffle(self.trainPatchesX, self.trainImagesX, self.trainY, random_state=self.seed)
        valPatchesX, valImagesX, valY = shuffle(self.valPatchesX, self.valImagesX, self.valY, random_state=self.seed)

        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(
            [trainPatchesX, trainImagesX], trainY,
            validation_data=([valPatchesX, valImagesX], valY),
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
        fig.savefig(currpath + "/etp_data/processed/figs/" + self.run_name + "train_val_acc.pdf", bbox_inches='tight')
        plt.show()
        # summarize history for loss
        fig = plt.figure(3)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        fig.savefig(currpath + "/etp_data/processed/figs/" + self.run_name + "train_val_loss.pdf", bbox_inches='tight')
        plt.show()

        # shuffle data
        testPatchesX, testImagesX, testY = shuffle(self.testPatchesX, self.testImagesX, self.testY, random_state=self.seed)


        # model evaluate
        results = self.model.evaluate([testPatchesX, testImagesX], testY, batch_size=128)
        print('test loss, test acc:', results)

        """
        # make predictions on the testing data
        predY = self.model.predict([testPatchesX, testImagesX]).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(testY, predY)

        auc_keras = auc(fpr_keras, tpr_keras)

        fig = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Area Under Roc = {:.3f}'.format(auc_keras))
        #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig.savefig(currpath + "/etp_data/processed/figs/" + self.run_name + "roc.pdf", bbox_inches='tight')
        plt.show()
        """