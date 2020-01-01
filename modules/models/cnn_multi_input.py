import os
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

import tensorflow as tf

from numpy.random import seed
seed(11)
from tensorflow import set_random_seed
set_random_seed(2)

path = os.getcwd()

use_gpu = tf.test.is_gpu_available()
print("use GPU?:",use_gpu )

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
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
	x = Dense(4)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

"""
def two_concat_cnn(activation='relu'):

    # Defining input 1
    input1 = Embedding(SAMPLE_SIZE, EMBEDDING_DIMS, input_length=MAX_SMI_LEN)
    x1 = Dropout(0.2)(input)
    x1 = Conv1D(NUM_FILTERS, FILTER_LENGTH, padding='valid', activation=activation, strides=1)(x1)
    x1 = GlobalMaxPooling1D()(x1)

    # Defining input 2
    input2 = Embedding(SAMPLE_SIZE, EMBEDDING_DIMS, input_length=MAX_SMI_LEN)
    x2 = Dropout(0.2)(input)
    x2 = Conv1D(NUM_FILTERS, FILTER_LENGTH, padding='valid', activation=activation, strides=1)(x2)
    x2 = GlobalMaxPooling1D()(x2)

    # Merging subnetworks
    x = concatenate([input1, input2])

    # Final Dense layer and compilation
    x = Dense(1, activation='sigmoid')
    model = Model(inputs=[input1, input2], x)
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])

    return model
"""