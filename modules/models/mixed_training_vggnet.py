# USAGE
# python mixed_training.py

# import the necessary packages
from modules.models import cnn_multi_input
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import yaml
from sklearn.utils import shuffle
from keras.layers.core import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import pandas as pd
from keras.layers.core import Dropout

import scipy.misc

expconfig = "../config/experimentconfig.yaml"
with open(expconfig, 'r') as ymlfile:
	cfg = yaml.load(ymlfile)
stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
datasetbuilder = DatasetBuilder()
print("Log.....Reading fixation data")
fixation_df = pd.read_pickle("../../etp_data/processed/fixation_df__40_subjects.pkl")
#choose stim to run on - Face or Snack
stim = stimFace
stimName = "face_imagenet"
fixation_specific_stim_df = fixation_df[fixation_df['stimType'] == stim.name]
fixation_specific_stim_df.reset_index(inplace=True)
img_size = (stim.size[0], stim.size[1])

#loading maps, images and labels datasets
try:
	print("loading maps")
	maps = np.load("../../etp_data/processed/maps.npy")
	print("loading images")
	images = np.load("../../etp_data/processed/images.npy")
	print("loading labels")
	labels = np.load("../../etp_data/processed/labels.npy")

except:
	maps = datasetbuilder.load_fixation_maps_dataset(fixation_specific_stim_df)
	images = datasetbuilder.load_images_dataset(fixation_specific_stim_df, img_size)
	labels = datasetbuilder.load_labels_dataset(fixation_specific_stim_df)
	maps, images, labels = shuffle(maps, images, labels, random_state=42)
	print("saving maps")
	np.save("../../etp_data/processed/" + stimName + "maps.npy", maps)
	print("saving images")
	np.save("../../etp_data/processed/" + stimName + "images.npy", images)
	print("saving labels")
	np.save("../../etp_data/processed/" + stimName + "labels.npy", labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
dataSize = len(maps)
trainSize = int(dataSize*0.75)
trainMapsX = maps[:trainSize]
testMapsX = maps[trainSize:]
trainImagesX = images[:trainSize]
testImagesX = images[trainSize:]
trainY = labels[:trainSize]
testY = labels[trainSize:]
#validation split
testDataSize = len(testMapsX)
testSize = int(testDataSize*0.75)
valMapsX = testMapsX[:testSize]
testMapsX = testMapsX[testSize:]
valImagesX = testImagesX[:testSize]
testImagesX = testImagesX[testSize:]
valY = testY[:testSize]
testY = testY[testSize:]


#split = train_test_split(maps, images, labels, test_size=0.25, random_state=42)
#(trainMapsX, testMapsX, trainImagesX, testImagesX, trainY, testY) = split

#split test data to test and validation
#validation_split = train_test_split(testMapsX, testImagesX, testY, test_size=0.25, random_state=42)
#(testMapsX, valMapsX, testImagesX, valImagesX, testY, valY) = validation_split

# create the two CNN models
vgg_map_model = cnn_multi_input.create_vggNet(stim.size[0], stim.size[1], 3)
vgg_image_model = cnn_multi_input.create_vggNet(stim.size[0], stim.size[1], 3)

for layer in vgg_map_model.layers:
	layer.name = layer.name + str("_map")
	layer.trainable = False

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

# our final model will accept scanpth on one CNN
# input and images on the second CNN input, outputting a single value as high or low bid (1/0)
model = Model(inputs=[vgg_map_model.input, vgg_image_model.input], outputs=x)
print(model.summary())

for layer in model.layers:
  print(layer.name)


# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss='binary_crossentropy', optimizer=opt,  metrics=['accuracy'])

# train the model
print("[INFO] training model...")
history = model.fit(
	[trainMapsX, trainImagesX], trainY,
	validation_data=([valMapsX, valImagesX], valY),
	epochs=100, batch_size=16)

# plot metrics
# summarize history for accuracy
fig = plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig("../../etp_data/processed/figs/" + stimName + "train_val_acc.pdf", bbox_inches='tight')
plt.show()
# summarize history for loss
fig = plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig.savefig("../../etp_data/processed/figs/" + stimName + "train_val_loss.pdf", bbox_inches='tight')
plt.show()

#model evaluate
results = model.evaluate([testMapsX, testImagesX], testY, batch_size=128)
print('test loss, test acc:', results)
# make predictions on the testing data
predY = model.predict([testMapsX, testImagesX]).ravel()
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
fig.savefig("../../etp_data/processed/figs/" + stimName + "roc.pdf", bbox_inches='tight')
plt.show()