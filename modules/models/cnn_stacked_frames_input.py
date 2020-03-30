import numpy
from keras.layers import Dense
from modules.models import cnn
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from keras.layers import concatenate
from keras.models import Model
import pandas as pd
import logging
from keras import optimizers

logger = logging.getLogger(__file__)


# CNN multi input
class CnnStackedFrames:
    def __init__(self, seed, dataset, saliency, run_name, patch_size, run_number, num_patches):
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
        self.batch_size = 64
        self.num_epochs = 60
        self.patch_size = patch_size
        self.num_class = 10
        self.num_patches = num_patches
        self.lr = 0.00001
        self.momentum = 0.9
        self.decay_rate = self.lr / self.num_epochs

    def define_model(self):
        # create the two CNN models
        cnn_map = cnn.cnn_for_image_concat(self.patch_size, self.patch_size, self.channel * self.num_patches)

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        x = Dense(self.num_class, activation="softmax")(cnn_map.output)

        # our final model will accept fixation map on one CNN
        # input and images on the second CNN input, outputting a single value as high or low bid (1/0)

        self.model = Model(inputs=cnn_map.input, outputs=x)

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
            trainMapsX, trainY,
            validation_data=(valMapsX, valY),
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
        results = self.model.evaluate(testMapsX, testY, batch_size=None)
        print('test loss, test acc:', results)
        results_df = pd.DataFrame(results, columns=[self.run_name + ", loss, acc"])
        results_df.to_csv(currpath + "/etp_data/processed/" + str(self.run_number) + "_results.csv", index=False)

        """
        # make predictions on the testing data
        predY = self.model.predict([testMapsX, testImagesX]).ravel()
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

        results_list = []
        results_list.append(results)
        results_list.append(auc_keras)
        results_df = pd.DataFrame(results_list)
        results_df.to_csv("cnn_multi_input_results_df.csv")
        """