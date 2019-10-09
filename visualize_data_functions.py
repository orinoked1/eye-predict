'''
@author: Dario Zanca, Ph.D. Student in Smart Computing
@institutions: University of Florence, University of Siena
@e-mail: dario.zanca@unifi.it
@tel: (+39) 333 82 78 072
@date: September, 2017
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ds_readers as get
import scipy.misc
import pandas as pd
import os
from eye_tracking_data_parser import raw_data_preprocess as parser


def map(FIXATION_MAP, imgToPlot_size, path, stimulus_name):
    """
    This functions visualize a specified stimulus adding the fixation map on top.
    """

    stimulus = get.stimulus(path, stimulus_name)

    toPlot = stimulus
    fixation_map = FIXATION_MAP
    fixation_map = cv2.cvtColor(np.uint8(fixation_map), cv2.COLOR_GRAY2RGB) * 255
    toPlot = cv2.resize(toPlot, imgToPlot_size)
    fin = cv2.addWeighted(fixation_map, 1, toPlot, 0.8, 0)

    scipy.misc.imsave('imageEX/'+'fixationMapEX.jpg', fin)

    return

def scanpath(SCANPATH, imgToPlot_size, path, stimulus_name, putNumbers = True, putLines = True, animation = True):

    """ This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.
        It is possible to visualize it as an animation by setting the additional
        argument animation=True.
       """

    stimulus = get.stimulus(path, stimulus_name)

    scanpath = SCANPATH

    #toPlot = [cv2.resize(stimulus, (520, 690)),] # look, it is a list!
    toPlot = [cv2.resize(stimulus, imgToPlot_size)]  # look, it is a list!

    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        cv2.circle(frame,
                   (fixation[1], fixation[2]),
                   5, (0, 0, 0), 1)
        if putNumbers:
            cv2.putText(frame, str(i+1),
                        (fixation[1], fixation[2]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), thickness=2)
        if putLines and i>0:
            prec_fixation = scanpath[i-1].astype(int)
            cv2.line(frame, (prec_fixation[1], prec_fixation[2]), (fixation[1], fixation[2]), (0, 0, 255), thickness = 1, lineType = 8, shift = 0)

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation: toPlot.pop(0)


    for i in range(len(toPlot)):
        if (i % 50) == 0:
            figName = str(i) + '_scanPathEX.jpg'
            scipy.misc.imsave('imageEX/'+figName, toPlot[i])

    return

def visualize(fixation_df, scanpath_df, stimType):

    path = '/Stim_0/'

    fixation_specific_stim_df = fixation_df[fixation_df['stimType'] == stimType]
    scanpath_specific_stim_df = scanpath_df[scanpath_df['stimType'] == stimType]

    fixation_sample = fixation_specific_stim_df.sample(n=1)

    sample = fixation_sample['sampleId'].values[0]
    print(sample)
    #y = os.getcwd()
    #raw_data_df = pd.read_csv(y + '/raw_data_01.csv')
    #parser.data_tidying(raw_data_df, [1080, 1920])
    #temp_df = raw_data_df[raw_data_df['subjectID']==int(sample[0:3])]
    #temp_df = temp_df[temp_df['stimName']==str(sample[4:])]

    sample_index = fixation_sample.index[0]
    scanpath_sample = scanpath_specific_stim_df.loc[sample_index]
    imgToPlot_size = (fixation_sample.fixationMap.values[0].shape[1], fixation_sample.fixationMap.values[0].shape[0])

    print('Log..... building fixation map')
    map(fixation_sample.fixationMap.values[0], imgToPlot_size, path, fixation_sample.stimName.values[0])
    print('Log... building scanpath')
    #scanpath(scanpath_sample.scanpath, imgToPlot_size, path, scanpath_sample.stimName, False)

    return