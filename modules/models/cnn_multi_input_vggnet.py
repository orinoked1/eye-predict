import numpy
from modules.models import cnn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers import concatenate
from keras.layers.core import Dropout


# CNN multi input
class CnnMultiInputVGGNet:
    def __init__(self, seed, dataset, saliency, run_name, stim_size):
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
        self.seed = seed
        self.batch_size = 16
        self.num_epochs = 2
        self.stimSize = stim_size

    def define_model(self):
        # create the two CNN models
        vgg_map_model = cnn.create_vggNet(self.stimSize[0], self.stimSize[1], self.channel)
        vgg_image_model = cnn.create_vggNet(self.stimSize[0],self.stimSize[1], self.channel)

        for layer in vgg_map_model.layers:
            layer.name = layer.name + str("_map")
        # layer.trainable = False

        for layer in vgg_image_model.layers:
            layer.name = layer.name + str("_image")
            layer.trainable = False

        # create the input to our final set of layers as the *output* of both CNNs
        combinedInput = concatenate([vgg_map_model.output, vgg_image_model.output])

        # Stacking a new simple convolutional network on top of it
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(combinedInput)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)

        # our final model will accept fixation map on one CNN
        # input and images on the second CNN input, outputting a single value as high or low bid (1/0)
        model = Model(inputs=[vgg_map_model.input, vgg_image_model.input], outputs=x)
        print(model.summary())

        for layer in model.layers:
            print(layer.name)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

        return

    def train_model(self):
        # shuffle data
        trainMapsX, trainImagesX, trainY = shuffle(self.trainMapsX, self.trainImagesX, self.trainY, random_state=self.seed)
        valMapsX, valImagesX, valY = shuffle(self.valMapsX, self.valImagesX, self.valY, random_state=self.seed)

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
        testMapsX, testImagesX, testY = shuffle(self.testMapsX, self.testImagesX, self.testY, random_state=self.seed)
        # shuffle data
        # testPatchesX, testY = shuffle(testPatchesX, testY, random_state=seed)

        # model evaluate
        results = self.model.evaluate([testMapsX, testImagesX], testY, batch_size=self.batch_size)
        print('test loss, test acc:', results)
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