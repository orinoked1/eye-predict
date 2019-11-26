import numpy as np
import cv2
import os
from eye_tracking_data_parser import raw_data_preprocess as parser
from sklearn.utils import shuffle
import pandas as pd
import pickle

class DatasetBuilder(object):

    def __init__(self, stimarray):
       self.stimarray = stimarray

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
                fixationMap = self.data_to_fixation_map_by_sampleId(data_df, sampleId, self.stimarray[0])
            else:
                fixationMap = self.data_to_fixation_map_by_sampleId(data_df, sampleId, self.stimarray[1])

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
        xedges = np.arange(stim.size[0])  # 480/576 TODO add the max image size as const
        yedges = np.arange(stim.size[1])  # 600/432

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
        # t = data_df[data_df['sampleId'] == sampleId].timeStamp.astype(int)
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
