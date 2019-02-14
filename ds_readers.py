import numpy as np
import cv2
import os
from eye_tracking_data_parser import raw_data_preprocess as parser
from sklearn.utils import shuffle
import pandas as pd
import pickle

COLLECTION_PATH = os.path.dirname(os.path.abspath(__file__)) + '/raw_data'

def get_raw_data():

    #read 'scale_ranking_bmm_short_data' row data into csv
    #TODO: read info from config file
    asc_files_path = '../raw_data/scale_ranking_bmm_short_data/output/asc'
    txt_files_path = '../raw_data/scale_ranking_bmm_short_data/output/txt'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'ScaleStart'
    csv_file_name = "scale_ranking_bmm_short_data.csv"
    scale_ranking_bmm_short_data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)

    #read 'bdm_bmm_short_data' row data into csv
    #TODO: read info from config file
    asc_files_path = '../raw_data/bdm_bmm_short_data/output/asc'
    txt_files_path = '../raw_data/bdm_bmm_short_data/output/txt'
    trial_satart_str = 'TrialStart'
    trial_end_str = 'Response'
    csv_file_name = "bdm_bmm_short_data.csv"
    bdm_bmm_short_data_csv_path = parser.raw_data_to_csv(asc_files_path, txt_files_path, trial_satart_str, trial_end_str, csv_file_name)


    # read csv into DF
    bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
    scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)

    return  bdm_bmm_short_data_df, scale_ranking_bmm_short_data_df

def get_all_data_df(bdm_bmm_short_data_csv_path, scale_ranking_bmm_short_data_csv_path):
    try:
        # read csv into DF
        bdm_bmm_short_data_df = pd.read_csv(bdm_bmm_short_data_csv_path)
        scale_ranking_bmm_short_data_df = pd.read_csv(scale_ranking_bmm_short_data_csv_path)
    except:
        print('ERROR: csv files does not exist! -> start running def to create it')
        bdm_bmm_short_data_df, scale_ranking_bmm_short_data_df = get_raw_data()

    # merge the to datasets to one
    data_df = pd.concat([bdm_bmm_short_data_df, scale_ranking_bmm_short_data_df])

    return data_df

def get_datasets_df():
    bdm_bmm_short_data_csv_path = 'eye_tracking_data_parser/bdm_bmm_short_data_df.csv'
    scale_ranking_bmm_short_data_csv_path = 'eye_tracking_data_parser/scale_ranking_bmm_short_data_df.csv'

    data_df = get_all_data_df(bdm_bmm_short_data_csv_path, scale_ranking_bmm_short_data_csv_path)

    try:
        with open('fixation_dataset_v1.pkl', 'rb') as f:
            print('Getting fixation dataset')
            fixation_dataset = pickle.load(f)
    except:
        print('ERROR: fixation dataset does not exist! -> start running def to create it')
        fixation_dataset = get_fixation_dataset(data_df, ([1080, 1920]))
        with open('fixation_dataset_v1.pkl', 'wb') as f:
            pickle.dump(fixation_dataset, f)

    fixation_df = pd.DataFrame(fixation_dataset)
    fixation_df.columns = ['stimName', 'stimType', 'sampleId', 'fixationMap', 'bid']

    try:
        with open('scanpath_dataset.pkl', 'rb') as f:
            print('Getting scanpath dataset')
            scanpath_dataset = pickle.load(f)
    except:
        print('ERROR: Scanpath dataset does not exist! -> start running def to create it')
        scanpath_dataset = get_scanpath_dataset(data_df, ([1080, 1920]))
        with open('scanpath_dataset.pkl', 'wb') as f:
            pickle.dump(scanpath_dataset, f)

    scanpath_df = pd.DataFrame(scanpath_dataset)
    scanpath_df.columns = ['stimName', 'stimType', 'sampleId', 'scanpath', 'bid']

    #hack for removing 999 bids (should have be done on the data tidying)
    fixation_df = fixation_df[fixation_df.bid != 999]
    scanpath_df = scanpath_df[scanpath_df.bid != 999]

    return fixation_df, scanpath_df

def stimulus(DATASET_NAME, STIMULUS_NAME):

    """ This functions returns the matrix of pixels of a specified stimulus.
        """

    path = COLLECTION_PATH + DATASET_NAME + STIMULUS_NAME
    image = cv2.imread(path, 1)

    return image


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

def get_train_test_dataset(data_df):
    y = data_df['bid']
    x = data_df['fixationMap']
    x, y = shuffle(x, y)
    # Allocating 80% of the data for train and 20% to test
    train_ratio = 0.8
    train_size = round(train_ratio * len(x))
    X_train, X_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return np.asanyarray(X_train), np.asanyarray(X_test), y_train, y_test

