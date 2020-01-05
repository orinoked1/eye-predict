# USAGE
# python mixed_training.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from modules.models import cnn_multi_input
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import yaml
from sklearn.utils import shuffle
import pickle
#from pyimagesearch import datasets
#from pyimagesearch import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os
import pandas as pd

expconfig = "../config/experimentconfig.yaml"
with open(expconfig, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
datasetbuilder = DatasetBuilder([stimSnack, stimFace])
print("Log.....Reading fixation data")
fixation_df = pd.read_pickle("../../etp_data/processed/fixation_df__40_subjects.pkl")
fixation_specific_stim_df = fixation_df[fixation_df['stimType'] == stimFace.name]
fixation_specific_stim_df.reset_index(inplace=True)
img_size = (stimFace.size[0], stimFace.size[1])

#loading maps, images and labels datasets
"""
maps = datasetbuilder.load_fixation_maps_dataset(fixation_specific_stim_df)
images = datasetbuilder.load_images_dataset(fixation_specific_stim_df, img_size)
labels = datasetbuilder.load_labels_dataset(fixation_specific_stim_df)
"""
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
	np.save("../../etp_data/processed/maps.npy", maps)
	print("saving images")
	np.save("../../etp_data/processed/images.npy", images)
	print("saving labels")
	np.save("../../etp_data/processed/labels.npy", labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
dataSize = len(maps)
trainSize = dataSize*0.75
trainMapsX = maps[:trainSize]
testMapsX = maps[trainSize:]
trainImagesX = images[:trainSize]
testImagesX = images[trainSize:]
trainY = labels[:trainSize]
testY = labels[trainSize:]
#validation split
testDataSize = len(testMapsX)
testSize = testDataSize*0.75
testMapsX = maps[:testSize]
valMapsX = maps[testSize:]
testImagesX = images[:testSize]
valImagesX = images[testSize:]
testY = labels[:testSize]
valY = labels[testSize:]

#split = train_test_split(maps, images, labels, test_size=0.25, random_state=42)
#(trainMapsX, testMapsX, trainImagesX, testImagesX, trainY, testY) = split

#split test data to test and validation
#validation_split = train_test_split(testMapsX, testImagesX, testY, test_size=0.25, random_state=42)
#(testMapsX, valMapsX, testImagesX, valImagesX, testY, valY) = validation_split

# create the two CNN models
cnn_scanpath = cnn_multi_input.create_cnn(600, 480, 3, regress=False)
cnn_image = cnn_multi_input.create_cnn(600, 480, 3, regress=False)

# create the input to our final set of layers as the *output* of both CNNs
combinedInput = concatenate([cnn_scanpath.output, cnn_image.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept scanpth on one CNN
# input and images on the second CNN input, outputting a single value as high or low bid (1/0)
model = Model(inputs=[cnn_scanpath.input, cnn_image.input], outputs=x)

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
	epochs=200, batch_size=8)

# plot metrics
plt.plot(history.history['acc'])
plt.show()

# make predictions on the testing data
predY = model.predict([testMapsX, testImagesX]).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testY, predY)

auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()


"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# load the house images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# process the house attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
(trainAttrX, testAttrX) = datasets.process_house_attributes(df,
	trainAttrX, testAttrX)

# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

# compile the model using mean absolute percentage error as our loss,
# implying that we seek to minimize the absolute percentage difference
# between our price *predictions* and the *actual prices*
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
	[trainAttrX, trainImagesX], trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
"""