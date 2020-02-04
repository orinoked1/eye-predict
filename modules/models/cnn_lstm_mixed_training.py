from modules.models import cnn_multi_input
from modules.data.datasets import DatasetBuilder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate


seed = 10
datasetbuilder = DatasetBuilder()
path = "../../etp_data/processed/scanpath_df__40_subjects.pkl"
stimType = "Face"
stimName = "face_sub_dist_"
scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(stimType, path)
patch_size = 60
df = datasetbuilder.create_patches_dataset(scanpaths, images, labels, patch_size, stim_size)
split = datasetbuilder.train_test_val_split_subjects_balnced_for_lstm(df, seed)
trainPatchesX, valPatchesX, testPatchesX, trainY, valY, testY = split
print("tt")
