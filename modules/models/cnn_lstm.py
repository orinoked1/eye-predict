import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from modules.models import cnn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt


# CNN LSTM for sequence classification
class CnnLstm:
    def __init__(self, seed, dataset, saliency, patch_size, run_name):
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

    def define_model(self):
        # define CNN model
        cn_net = cnn.create_cnn(self.patch_size, self.patch_size, self.channel, regress=False)
        # define time distributer
        # define LSTM model
        self.model = Sequential()
        self.model.add(TimeDistributed(cn_net, input_shape=(50, self.patch_size, self.patch_size, self.channel)))
        self.model.add(LSTM(10, activation='relu', return_sequences=False))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())
        return

    def train_model(self):
        # shuffle data
        print("[INFO] Shuffle dataset...")
        trainPatchesX, trainY = shuffle(self.trainPatchesX, self.trainY, random_state=self.seed)
        valPatchesX, valY = shuffle(self.valPatchesX,  self.valY, random_state=self.seed)

        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(trainPatchesX, trainY,
            validation_data=(valPatchesX, valY),
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
        testPatchesX, testY = shuffle(self.testPatchesX, self.testY, random_state=self.seed)

        #model evaluate
        results = self.model.evaluate(testPatchesX, testY, batch_size=self.batch_size)
        print('test loss, test acc:', results)
        # make predictions on the testing data
        predY = self.model.predict(testPatchesX).ravel()
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