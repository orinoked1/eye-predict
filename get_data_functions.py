import numpy as np
import cv2
import os
from eye_tracking_data_parser import raw_data_preprocess as parser


COLLECTION_PATH = os.path.dirname(os.path.abspath(__file__)) + '/raw_data'

def stimulus(DATASET_NAME, STIMULUS_NAME):

    ''' This functions returns the matrix of pixels of a specified stimulus.
        Notice that, of course, both DATASET_NAME and STIMULUS_NAME need
        to be specified. The latter, must include file extension.
        The returned matrix could be 2- or 3-dimesional. '''

    return cv2.imread(COLLECTION_PATH+'/'
                      +DATASET_NAME
                      +STIMULUS_NAME, 1)


def data_to_fixation_map_by_sampleId(data_df, sampleId):
    print('Log..... get fixation map for sampleId: ', sampleId)
    x = data_df[data_df['sampleId'] == sampleId].X_axis
    y = data_df[data_df['sampleId'] == sampleId].Y_axis
    if len(x) | len(y) < 5:
        print('Fixation data is None for sampleId: ', sampleId)
        return None
    xedges = np.arange(576)
    yedges = np.arange(432)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    heatmap = heatmap.T  # Let each row list bins with common y range.

    """
    plt.imshow(heatmap)
    plt.show()

    matrix = np.zeros((433, 577))
    temp = [x,y]
    temp = np.asanyarray(temp).astype(int)
    tt = np.transpose(temp)
    for t in tt:
        print(t)
        matrix[t[1],t[0]] = 1.

    plt.imshow(matrix)
    plt.show()
    """
    return heatmap

def get_fixation_dataset(data_df, screen_resolution):
    data_df = parser.data_tidying(data_df, screen_resolution)
    print('Log..... Build fixation dataset')
    fixation_dataset = []
    for sampleId in data_df.sampleId.unique():
        sample_data = []
        bid = data_df[data_df['sampleId'] == sampleId].bid.unique()
        stimName = data_df[data_df['sampleId'] == sampleId].stimName.unique()
        stimType = data_df[data_df['sampleId'] == sampleId].stimType.unique()
        sample = data_df[data_df['sampleId'] == sampleId].sampleId.unique()
        fixationMap = data_to_fixation_map_by_sampleId(data_df, sampleId)
        if type(fixationMap) is not np.ndarray:
            continue
        sample_data.append(stimName[0])
        sample_data.append(stimType[0])
        sample_data.append(sample[0])
        sample_data.append(fixationMap)
        sample_data.append(bid[0])
        fixation_dataset.append(sample_data)

    return fixation_dataset

def define_timeStamp_downSampling():
    return

def data_to_scanpath(data_df, sampleId, downsamplemillisec):
    print('Log..... get scanpath data for sampleId: ', sampleId)
    #todo - add downsampling option for the data poinnts
    scanpath = []
    t = data_df[data_df['sampleId'] == sampleId].timeStamp
    x = data_df[data_df['sampleId'] == sampleId].X_axis
    y = data_df[data_df['sampleId'] == sampleId].Y_axis
    if len(x) | len(y) < 5:
        print('Scanpath data is None for sampleId: ', sampleId)
        return None
    scanpath.append(t)
    scanpath.append(x)
    scanpath.append(y)

    scanpath = np.asanyarray(scanpath).T

    return np.asanyarray(scanpath)

def get_scanpath_dataset(data_df, screen_resolution, downsamplemillisec = 4):
    data_df = parser.data_tidying(data_df, screen_resolution)
    print('Log..... Build scanpath dataset')
    scanpath_dataset = []
    for sampleId in data_df.sampleId.unique():
        sample_data = []
        bid = data_df[data_df['sampleId'] == sampleId].bid.unique()
        stimName = data_df[data_df['sampleId'] == sampleId].stimName.unique()
        stimType = data_df[data_df['sampleId'] == sampleId].stimType.unique()
        sample = data_df[data_df['sampleId'] == sampleId].sampleId.unique()
        scanpath = data_to_scanpath(data_df, sampleId, downsamplemillisec)
        if type(scanpath) is not np.ndarray:
            continue
        sample_data.append(stimName[0])
        sample_data.append(stimType[0])
        sample_data.append(sample[0])
        sample_data.append(scanpath)
        sample_data.append(bid[0])
        scanpath_dataset.append(sample_data)

    return scanpath_dataset