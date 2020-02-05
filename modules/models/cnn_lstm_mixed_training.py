from modules.data.datasets import DatasetBuilder
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from modules.models import cnn_multi_input
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt


seed = 33
datasetbuilder = DatasetBuilder()
path = "../../etp_data/processed/scanpath_df__40_subjects.pkl"
stimType = "Face"
stimName = "face_lstm_3_seed_33"
scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(stimType, path)
patch_size = 60
saliency=False
if saliency:
    channel = 1
else:
    channel = 3
df = datasetbuilder.create_patches_dataset(scanpaths, images, labels, patch_size, saliency)
split = datasetbuilder.train_test_val_split_subjects_balnced_for_lstm(df, seed)
trainPatchesX, valPatchesX, testPatchesX, trainY, valY, testY = split


# CNN LSTM for sequence classification
# fix random seed for reproducibility
numpy.random.seed(seed)
# define CNN model
cnn = cnn_multi_input.create_cnn(patch_size, patch_size, channel, regress=False)
# define time distributer
# define LSTM model
model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(50, patch_size, patch_size, channel)))
model.add(LSTM(10, activation='relu', return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# shuffle data
trainPatchesX, trainY = shuffle(trainPatchesX, trainY, random_state=seed)
valPatchesX, valY = shuffle(valPatchesX,  valY, random_state=seed)

# train the model
print("[INFO] training model...")
if saliency:
    trainPatchesX = numpy.reshape(trainPatchesX, (trainPatchesX.shape, -1))
history = model.fit(trainPatchesX, trainY,
	validation_data=(valPatchesX, valY),
	epochs=20, batch_size=16)

# plot metrics
# summarize history for accuracy
fig = plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig("../../etp_data/processed/figs/" + stimName + "train_val_acc.pdf", bbox_inches='tight')
plt.show()
# summarize history for loss
fig = plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig("../../etp_data/processed/figs/" + stimName + "train_val_loss.pdf", bbox_inches='tight')
plt.show()

# shuffle data
testPatchesX, testY = shuffle(testPatchesX, testY, random_state=seed)

#model evaluate
results = model.evaluate(testPatchesX, testY, batch_size=128)
print('test loss, test acc:', results)
# make predictions on the testing data
predY = model.predict(testPatchesX).ravel()
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