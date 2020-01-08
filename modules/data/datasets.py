import numpy as np
import cv2
import os
from eye_tracking_data_parser import raw_data_preprocess as parser
from sklearn.utils import shuffle
import pandas as pd
from modules.data.visualization import DataVis
import pickle
import yaml
from modules.data.stim import Stim


class DatasetBuilder(object):

    def __init__(self):
        expconfig = "../config/experimentconfig.yaml"
        with open(expconfig, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'],
                         cfg['exp']['etp']['stimSnack']['size'])
        self.stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'],
                        cfg['exp']['etp']['stimFace']['size'])

    def get_fixation_dataset(self, data_df):
        print('Log..... Build fixation dataset')
        fixation_dataset = []
        for sampleId in data_df.sampleId.unique():
            sample_data = []
            bid = data_df[data_df['sampleId'] == sampleId].bid.unique()
            stimName = data_df[data_df['sampleId'] == sampleId].stimName.unique()
            stimType = data_df[data_df['sampleId'] == sampleId].stimType.unique()
            sample = data_df[data_df['sampleId'] == sampleId].sampleId.unique()
            if stimType == 'Snack':
                fixationMap = self.data_to_fixation_map_by_sampleId(data_df, sampleId, self.stimSnack)
            else:
                fixationMap = self.data_to_fixation_map_by_sampleId(data_df, sampleId, self.stimFace)

            if type(fixationMap) is not np.ndarray:
                continue
            sample_data.append(stimName[0])
            sample_data.append(stimType[0])
            sample_data.append(sample[0])
            sample_data.append(fixationMap)
            sample_data.append(bid[0])
            fixation_dataset.append(sample_data)

        fixation_df = pd.DataFrame(fixation_dataset)
        fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']

        return fixation_df

    def data_to_fixation_map_by_sampleId(self, data_df, sampleId, stim):
        print('Log..... get fixation map for sampleId: ', sampleId)
        x = data_df[data_df['sampleId'] == sampleId].X_axis
        y = data_df[data_df['sampleId'] == sampleId].Y_axis
        if len(x) | len(y) < 5:
            print('Fixation data is None for sampleId: ', sampleId)
            return None
        xedges = np.arange(stim.size[0])
        yedges = np.arange(stim.size[1])

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        heatmap = heatmap.T  # Let each row list bins with common y range.

        return heatmap

    def get_scanpath_dataset(self, data_df):
        print('Log..... Build scanpath dataset')
        scanpath_dataset = []
        for sampleId in data_df.sampleId.unique():
            sample_data = []
            bid = data_df[data_df['sampleId'] == sampleId].bid.unique()
            #subjectID = data_df[data_df['sampleId'] == sampleId].subjectID.unique()
            stimName = data_df[data_df['sampleId'] == sampleId].stimName.unique()
            stimType = data_df[data_df['sampleId'] == sampleId].stimType.unique()
            sample = data_df[data_df['sampleId'] == sampleId].sampleId.unique()
            scanpath = self.data_to_scanpath(data_df, sampleId)
            if type(scanpath) is not np.ndarray:
                continue
            #sample_data.append(subjectID[0])
            sample_data.append(stimName[0])
            sample_data.append(stimType[0])
            sample_data.append(sample[0])
            sample_data.append(scanpath)
            sample_data.append(bid[0])
            scanpath_dataset.append(sample_data)

        scanpath_df = pd.DataFrame(scanpath_dataset)
        scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

        return scanpath_df

    def data_to_scanpath(self, data_df, sampleId):
        print('Log..... get scanpath data for sampleId: ', sampleId)
        # todo - add downsampling option for the data poinnts
        scanpath = []
        t = data_df[data_df['sampleId'] == sampleId].timeStamp.astype(int)
        x = data_df[data_df['sampleId'] == sampleId].X_axis.astype(int)
        y = data_df[data_df['sampleId'] == sampleId].Y_axis.astype(int)
        if len(x) | len(y) < 5:
            print('Scanpath data is None for sampleId: ', sampleId)
            return None
        # scanpath.append(t)
        scanpath.append(x)
        scanpath.append(y)

        scanpath = np.asanyarray(scanpath).T

        return np.asanyarray(scanpath)

    def load_fixation_maps_dataset(self, df):
        print("Log.....Loading maps")
        maps = []
        for fixation_map in np.asanyarray(df.fixationMap):
            print("....")
            fixation_map = np.pad(fixation_map, [(0, 1), (0, 1)], mode='constant')
            fixation_map = cv2.cvtColor(np.uint8(fixation_map), cv2.COLOR_GRAY2RGB) * 255
            fixation_map = fixation_map/255
            maps.append(fixation_map)

        return maps

    def load_images_dataset(self, df, img_size):
        print("Log.....Loading images")
        images = []
        for image in np.asanyarray(df.stimName):
            print("....")
            img = DataVis.stimulus("../../etp_data/Stim_0/", image)
            img = cv2.resize(img, img_size)
            img = img / 255
            images.append(img)

        return images

    def load_labels_dataset(self, df):
        print("Log.....Loading labels")
        df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])
        labels = np.asanyarray(df.binary_bid)

        return labels

    def load_fixations_related_datasets(self, stimType, path):
        print("Log.....Reading fixation data")
        fixation_df = pd.read_pickle(path)
        # choose stim to run on - Face or Snack
        if stimType == "Face":
            stim = self.stimFace
            stimName = "face_imagenet_"
        else:
            stim = self.stimSnack
            stimName = "snack_imagenet_"
        fixation_df_by_stim = fixation_df[fixation_df['stimType'] == stim.name]
        fixation_df_by_stim.reset_index(inplace=True)
        stim_size = (stim.size[0], stim.size[1])

        maps = self.load_fixation_maps_dataset(fixation_df_by_stim)
        images = self.load_images_dataset(fixation_df_by_stim, stim_size)
        labels = self.load_labels_dataset(fixation_df_by_stim)

        return maps, images, labels, stim_size

    def train_val_test_split_per_subject(self, maps, images, labels):



        return trainMapsX, valMapsX, testMapsX, trainImagesX, testImagesX, valImagesX, trainY, valY, testY