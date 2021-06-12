import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import sequence


# SIMPLE LSTM for sequence classification
class SimpleLstm:
    def __init__(self, seed, dataset, run_name):
        # fix random seed for reproducibility
        numpy.random.seed(seed)
        self.trainScanpathX, self.valScanpathX, self.testScanpathX, \
        self.trainImagesX, self.valImagesX, self.testImagesX, self.trainY, self.valY, self.testY = dataset
        self.model = None
        self.history = None
        self.run_name = run_name
        self.seed = seed
        self.batch_size = 16
        self.num_epochs = 10
        self.max_review_length = 1500
        self.trainScanpathX = sequence.pad_sequences(self.trainScanpathX, maxlen=self.max_review_length)
        self.valScanpathX = sequence.pad_sequences(self.valScanpathX,
                                                     maxlen=self.max_review_length)
        self.testScanpathX = sequence.pad_sequences(self.testScanpathX,
                                                     maxlen=self.max_review_length)

    def define_model(self):
        self.model = Sequential()
        self.model.add(LSTM(10, input_shape=(self.max_review_length, 2)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

        return


    def train_model(self):
        # shuffle data
        print("[INFO] Shuffle dataset...")
        trainScanpathX, trainY = shuffle(self.trainScanpathX, self.trainY, random_state=self.seed)
        valScanpathX, valY = shuffle(self.valScanpathX,  self.valY, random_state=self.seed)

        # train the model
        print("[INFO] training model...")
        self.history = self.model.fit(trainScanpathX, trainY,
            validation_data=(valScanpathX, valY),
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
        testScanpathX, testY = shuffle(self.testScanpathX, self.testY, random_state=self.seed)

        #model evaluate
        results = self.model.evaluate(testScanpathX, testY, batch_size=self.batch_size)
        print('test loss, test acc:', results)
        # make predictions on the testing data
        predY = self.model.predict(testScanpathX).ravel()
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