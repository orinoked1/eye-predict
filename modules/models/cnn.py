import os
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras import initializers

import tensorflow as tf
"""
from numpy.random import seed
seed(11)
from tensorflow import set_random_seed
set_random_seed(2)
"""
path = os.getcwd()

use_gpu = tf.test.is_gpu_available()
print("use GPU?:",use_gpu )

def cnn_for_image_concat(width, height, depth):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (width, height, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)
	x = inputs
	x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
	x = BatchNormalization(axis=-1)(x)
	x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
	x = Flatten()(x)
	x = Dense(units=128, activation="relu")(x)
	x = Dropout(0.3)(x)
	x = Dense(units=32, activation="relu")(x)
	x = Dropout(0.3)(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model


def create_cnn(width, height, depth, filters=(64, 32, 16)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding=2)(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
#	x = Dense(4)(x)
#	x = Activation("relu")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

def fixation_cnn(width, height, depth, filters=(16, 32, 64)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	#	x = Dense(4)(x)
	#	x = Activation("relu")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

def map_vggNet(width, height, depth, filters=(16, 32, 64)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	weightInit = initializers.RandomNormal(stddev=0.10, seed=None)

	# define the model input
	inputs = Input(shape=inputShape)

	base_model = VGG16(weights='imagenet', include_top=False, input_shape=inputShape)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(512, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
	x = Dense(128, activation='relu')(x)  # dense layer 3
	x = BatchNormalization(axis=chanDim)(x)
	#x = Dropout(0.5)(x)

	# construct the CNN
	model = Model(base_model.input, x)

	for layer in base_model.layers:
		layer.name = layer.name + str("_map")
		layer.trainable = False
	#for layer in model.layers[:7]:
	#	layer.trainable = False


	# return the CNN
	return model

def image_vggNet(width, height, depth, filters=(16, 32, 64)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	weightInit = initializers.RandomNormal(stddev=0.10, seed=None)

	# define the model input
	inputs = Input(shape=inputShape)

	base_model = VGG16(weights='imagenet', include_top=False, input_shape=inputShape)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	#x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
	x = Dense(512, activation='relu')(x)  # dense layer 3
	x = Dense(128, activation='relu')(x)
	x = BatchNormalization(axis=chanDim)(x)
	#x = Dropout(0.5)(x)

	# construct the CNN
	model = Model(base_model.input, x)

	for layer in model.layers:
		layer.name = layer.name + str("_image")
		layer.trainable = False
	#for layer in model.layers[:7]:
	#	layer.trainable = False


	# return the CNN
	return model

def image_resNet(width, height, depth, filters=(16, 32, 64)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=inputShape)
	x = base_model.output
	#x = GlobalAveragePooling2D()(x)
	#x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
	#x = Dense(512, activation='relu')(x)  # dense layer 3
	#x = BatchNormalization(axis=chanDim)(x)
	#x = Dropout(0.5)(x)

	# construct the CNN
	model = Model(base_model.input, x)

	for layer in model.layers:
		layer.name = layer.name + str("_image")
		#layer.trainable = False
	model.layers[0].trainable = False

	# return the CNN
	return model

def map_resNet(width, height, depth, filters=(16, 32, 64)): #(16, 32, 64, 128)
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=inputShape)
	x = base_model.output
	#x = GlobalAveragePooling2D()(x)
	#x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
	#x = Dense(512, activation='relu')(x)  # dense layer 3
	#x = BatchNormalization(axis=chanDim)(x)
	#x = Dropout(0.5)(x)

	# construct the CNN
	model = Model(base_model.input, x)

	for layer in model.layers:
		layer.name = layer.name + str("_map")
		#layer.trainable = False
	model.layers[0].trainable = False

	# return the CNN
	return model

def create_vggNet(width, height, depth):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)

	# define the model input
	inputs = Input(shape=inputShape)

	model = VGG16(include_top=False, weights='imagenet', input_shape=inputShape)

	return model

def lstm():
	return