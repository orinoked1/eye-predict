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
from sklearn.utils import shuffle
import scipy.misc
from sklearn.cluster import KMeans


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
        df['fixationMap'] = df['fixationMap'].apply(lambda x: np.pad(x, [(0, 1), (0, 1)], mode='constant'))
        df['fixationMap'] = df['fixationMap'].apply(lambda x: cv2.cvtColor(np.uint8(x), cv2.COLOR_GRAY2RGB) * 255)
        df['fixationMap'] = df['fixationMap'].apply(lambda x: x / 255)

        return df[["sampleId", "fixationMap"]]

    def load_scanpath_dataset(self, df):
        print("Log.....Loading scanpaths")
        return df[["sampleId", "scanpath"]]

    def load_images_dataset(self, df, img_size):
        print("Log.....Loading images")
        img_dict = {}
        for image in np.asanyarray(df.stimName.unique()):
            print("loading image - " + image)
            img = DataVis.stimulus("../../etp_data/Stim_0/", image)
            img = cv2.resize(img, img_size)
            img = img / 255
            img_dict[image] = img
        img_df = pd.DataFrame(list(img_dict.items()), columns=['stimName', 'img'])
        #scipy.misc.imsave("../../etp_data/processed/temp0.jpg", img_dict["1_1027.jpg"])
        newdf = pd.merge(df, img_df, on='stimName', how='left')
        #x = dfnew[dfnew['stimName'] == "1_1027.jpg"]
        #scipy.misc.imsave("../../etp_data/processed/temp1.jpg", x["img"].values[0])

        return newdf[["sampleId", "img"]]

    def load_images_for_scanpath_dataset(self, df, img_size):
        print("Log.....Loading images")
        img_dict = {}
        for image in np.asanyarray(df.stimName.unique()):
            print("loading image - " + image)
            img = DataVis.stimulus("../../etp_data/Stim_0/", image)
            img = cv2.resize(img, img_size)
            img_dict[image] = img
        img_df = pd.DataFrame(list(img_dict.items()), columns=['stimName', 'img'])
        #scipy.misc.imsave("../../etp_data/processed/temp0.jpg", img_dict["1_1027.jpg"])
        newdf = pd.merge(df, img_df, on='stimName', how='left')
        #x = dfnew[dfnew['stimName'] == "1_1027.jpg"]
        #scipy.misc.imsave("../../etp_data/processed/temp1.jpg", x["img"].values[0])

        return newdf[["sampleId", "img"]]

    def load_labels_dataset(self, df):
        print("Log.....Loading labels")
        df['binary_bid'] = pd.qcut(df.bid, 2, labels=[0, 1])
        #labels = np.asanyarray(df.binary_bid)

        return df[["sampleId", "binary_bid"]]

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

    def load_scanpath_related_datasets(self, stimType, path):
        print("Log.....Reading scanpath data")
        scanpath_df = pd.read_pickle(path)
        # choose stim to run on - Face or Snack
        if stimType == "Face":
            stim = self.stimFace
            stimName = "face_imagenet_"
        else:
            stim = self.stimSnack
            stimName = "snack_imagenet_"
        scanpath_df_by_stim = scanpath_df[scanpath_df['stimType'] == stim.name]
        scanpath_df_by_stim.reset_index(inplace=True)
        stim_size = (stim.size[0], stim.size[1])

        scanpaths = self.load_scanpath_dataset(scanpath_df_by_stim)
        images = self.load_images_for_scanpath_dataset(scanpath_df_by_stim, stim_size)
        labels = self.load_labels_dataset(scanpath_df_by_stim)

        return scanpaths, images, labels, stim_size

    def train_test_val_split_subjects_balnced(self, df, seed, is_patch):

        df["subjectId"] = df['sampleId'].apply(lambda x: x.split("_")[0])
        trainset = []
        testset = []
        valset = []
        flag = 0
        for subject in df.subjectId.unique():
            dfsubject = df[df["subjectId"] == subject]
            dfsubject = shuffle(dfsubject, random_state=seed)
            dataSize = dfsubject.shape[0]
            trainSize = int(dataSize * 0.75)
            train = dfsubject[:trainSize]
            test = dfsubject[trainSize:]
            # validation split
            testDataSize = test.shape[0]
            testSize = int(testDataSize * 0.70)
            val = test[:testSize]
            test = test[testSize:]
            if flag == 0:
                trainset = train
                valset = val
                testset = test
                flag = 1
            else:
                trainset = pd.concat([trainset, train])
                valset = pd.concat([valset, val])
                testset = pd.concat([testset, test])

        print("train", trainset.binary_bid.value_counts())
        print("val", valset.binary_bid.value_counts())
        print("test", testset.binary_bid.value_counts())

        print("Building train, val, test datasets...")
        if is_patch:
            trainMapsX = np.asanyarray(trainset.patch.tolist())
            valMapsX = np.asanyarray(valset.patch.tolist())
            testMapsX = np.asanyarray(testset.patch.tolist())
        else:
            trainMapsX = np.asanyarray(trainset.fixationMap.tolist())
            valMapsX = np.asanyarray(valset.fixationMap.tolist())
            testMapsX = np.asanyarray(testset.fixationMap.tolist())
        trainImagesX = np.asanyarray(trainset.img.tolist())
        valImagesX = np.asanyarray(valset.img.tolist())
        testImagesX = np.asanyarray(testset.img.tolist())
        trainY = np.asanyarray(trainset.binary_bid.tolist())
        valY = np.asanyarray(valset.binary_bid.tolist())
        testY = np.asanyarray(testset.binary_bid.tolist())

        return trainMapsX, valMapsX, testMapsX, trainImagesX, valImagesX, testImagesX, trainY, valY, testY

    def train_test_val_split_subjects_balnced_for_lstm(self, df, seed):

        df["subjectId"] = df['sampleId'].apply(lambda x: x.split("_")[0])
        trainset = []
        testset = []
        valset = []
        flag = 0
        for subject in df.subjectId.unique():
            dfsubject = df[df["subjectId"] == subject]
            dfsubject = shuffle(dfsubject, random_state=seed)
            dataSize = dfsubject.shape[0]
            trainSize = int(dataSize * 0.75)
            train = dfsubject[:trainSize]
            test = dfsubject[trainSize:]
            # validation split
            testDataSize = test.shape[0]
            testSize = int(testDataSize * 0.70)
            val = test[:testSize]
            test = test[testSize:]
            if flag == 0:
                trainset = train
                valset = val
                testset = test
                flag = 1
            else:
                trainset = pd.concat([trainset, train])
                valset = pd.concat([valset, val])
                testset = pd.concat([testset, test])

        #print("train", trainset.binary_bid.value_counts())
        #print("val", valset.binary_bid.value_counts())
        #print("test", testset.binary_bid.value_counts())

        print("Building train, val, test datasets...")
        trainPatchesX = np.asanyarray(trainset.patch.tolist())
        valPatchesX = np.asanyarray(valset.patch.tolist())
        testPatchesX = np.asanyarray(testset.patch.tolist())
        trainY = np.asanyarray(trainset.binary_bid.tolist())
        valY = np.asanyarray(valset.binary_bid.tolist())
        testY = np.asanyarray(testset.binary_bid.tolist())

        return trainPatchesX, valPatchesX, testPatchesX, trainY, valY, testY


    def create_patches_dataset(self, scanpaths, images, labels, patch_size, saliency):
        print("Log.....Building patches")
        df = scanpaths.merge(images,on='sampleId').merge(labels,on='sampleId')
        df['scanpath_len'] = 0
        for i in range(df.scanpath.size):
            df.at[i, 'scanpath_len'] = len(df.scanpath[i])
        # prepering the data
        df = df[df.scanpath_len > 2300]  # > 85%
        df = df.reset_index()
        patches_list = []
        for scanpath, img in zip(df.scanpath, df.img):
            scipy.misc.imsave("../../etp_data/processed/patches/" + "original_img.jpg", img)
            if saliency:
                # initialize OpenCV's static saliency spectral residual detector and
                # compute the saliency map
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, img) = saliency.computeSaliency(img)
                img = (img * 255).astype("uint8")
                #scipy.misc.imsave("../../etp_data/processed/patches/" + "saliency_img.jpg", img)
            # Compute clusters Means through time
            fixations_centers = []
            clusters = np.array_split(scanpath, 50)
            for i in clusters:
                center = np.mean(i, axis=0).round().astype(int)
                fixations_centers.append(center)
            #build patches around the fixations_centers
            patches = []
            patch_num = 1
            #DataVis.scanpath_by_img("../../etp_data/processed/patches/", scanpath, stim_size, img, False)
            for xi, yi in fixations_centers:
                length = int(patch_size/2)
                patch = img[yi - length: yi + length, xi - length: xi + length]
                padx = 60 - patch.shape[0]
                pady = 60 - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, padx, 0, pady, cv2.BORDER_CONSTANT)
                #scipy.misc.imsave("../../etp_data/processed/patches/" + str(patch_num) + "_patch_ORG.jpg", patch)
                patch = patch/255
                patch = patch[:, :, np.newaxis]
                patches.append(patch)
                patch_num += 1
            patches_list.append(np.asanyarray(patches))
            img = (img / 255).astype("uint8")

        df["patch"] = patches_list
        df = df[["sampleId", "patch", "img", "binary_bid"]]

        return df