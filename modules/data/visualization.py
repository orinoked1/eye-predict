import cv2
import numpy as np
import os
import scipy.misc
#from heatmappy import Heatmapper

class DataVis(object):

    def __init__(self, stimpath, vispath, stimarray, stiminx):
        self.stimarray = stimarray
        self.stim = self.stimarray[stiminx]
        self.stimpath = stimpath
        self.vispath = vispath
        self.currpath = os.getcwd()

    @staticmethod
    def stimulus(DATASET_NAME, STIMULUS_NAME):

        """ This functions returns the matrix of pixels of a specified stimulus.
            """
        path = DATASET_NAME + STIMULUS_NAME
        image = cv2.imread(path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def heatmap(self, SCANPATH, imgToPlot_size, path, stimulus_name, dest_fn):
        stimulus = self.stimulus(path, stimulus_name)
        stimulus = cv2.resize(stimulus, imgToPlot_size)
        example_points = SCANPATH
        example_img = stimulus
        heatmapper = Heatmapper()
        heatmap = heatmapper.heatmap_on_img(example_points, example_img)

        scipy.misc.imsave(self.currpath + self.vispath + 'heatMapEX.jpg', heatmap)

        return

    def map(self, FIXATION_MAP, imgToPlot_size, path, stimulus_name):
        """
        This functions visualize a specified stimulus adding the fixation map on top.
        """

        stimulus = self.stimulus(path, stimulus_name)

        toPlot = stimulus
        fixation_map = FIXATION_MAP
        fixation_map = np.pad(fixation_map, [(0, 1), (0, 1)], mode='constant')
        fixation_map = cv2.cvtColor(np.uint8(fixation_map), cv2.COLOR_GRAY2RGB) * 255
        toPlot = cv2.resize(toPlot, imgToPlot_size)
        fin = cv2.addWeighted(fixation_map, 1, toPlot, 0.8, 0)

        scipy.misc.imsave(self.currpath + self.vispath + 'fixationMapEX.jpg', fin)

        return

    def scanpath(self, SCANPATH, imgToPlot_size, path, stimulus_name, putNumbers=True, putLines=True, animation=True):

        """ This functions uses cv2 standard library to visualize the scanpath
            of a specified stimulus.
            It is possible to visualize it as an animation by setting the additional
            argument animation=True.
           """

        stimulus = self.stimulus(path, stimulus_name)

        scanpath = SCANPATH

        # toPlot = [cv2.resize(stimulus, (520, 690)),] # look, it is a list!
        toPlot = [cv2.resize(stimulus, imgToPlot_size)]  # look, it is a list!

        for i in range(np.shape(scanpath)[0]):

            fixation = scanpath[i].astype(int)

            frame = np.copy(toPlot[-1]).astype(np.uint8)

            cv2.circle(frame,
                       (fixation[0], fixation[1]),
                       5, (0, 0, 0), 1)
            if putNumbers:
                cv2.putText(frame, str(i + 1),
                            (fixation[0], fixation[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), thickness=2)
            if putLines and i > 0:
                prec_fixation = scanpath[i - 1].astype(int)
                cv2.line(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]), (0, 0, 255),
                         thickness=1, lineType=8, shift=0)

            # if animation is required, frames are attached in a sequence
            # if not animation is required, older frames are removed
            toPlot.append(frame)
            if not animation: toPlot.pop(0)

        for i in range(len(toPlot)):
            if (i % 50) == 0:
                figName = str(i) + '_scanPathEX.jpg'
                scipy.misc.imsave(self.currpath + self.vispath + figName, toPlot[i])

        return

    @staticmethod
    def scanpath_by_img(path, SCANPATH, imgToPlot_size, stimulus, putNumbers=True, putLines=True, animation=True):

        """ This functions uses cv2 standard library to visualize the scanpath
            of a specified stimulus.
            It is possible to visualize it as an animation by setting the additional
            argument animation=True.
           """

        scanpath = SCANPATH

        # toPlot = [cv2.resize(stimulus, (520, 690)),] # look, it is a list!
        toPlot = [cv2.resize(stimulus, imgToPlot_size)]  # look, it is a list!

        for i in range(np.shape(scanpath)[0]):

            fixation = scanpath[i].astype(int)

            frame = np.copy(toPlot[-1]).astype(np.uint8)

            cv2.circle(frame,
                       (fixation[0], fixation[1]),
                       5, (0, 0, 0), 1)
            if putNumbers:
                cv2.putText(frame, str(i + 1),
                            (fixation[0], fixation[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), thickness=2)
            if putLines and i > 0:
                prec_fixation = scanpath[i - 1].astype(int)
                cv2.line(frame, (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]), (0, 0, 255),
                         thickness=1, lineType=8, shift=0)

            # if animation is required, frames are attached in a sequence
            # if not animation is required, older frames are removed
            toPlot.append(frame)
            if not animation: toPlot.pop(0)

        for i in range(len(toPlot)):
            if (i % 50) == 0:
                figName = str(i) + '_scanPathEX.jpg'
                scipy.misc.imsave(path + figName, toPlot[i])

        return

    def visualize(self, fixation_df, scanpath_df):

        fixation_specific_stim_df = fixation_df[fixation_df['stimType'] == self.stim.name]
        scanpath_specific_stim_df = scanpath_df[scanpath_df['stimType'] == self.stim.name]

        np.random.seed(404)
        fixation_sample = fixation_specific_stim_df.sample(n=1)
        f_stimName = fixation_sample.stimName.values[0]
        sample = fixation_sample['sampleId'].values[0]
        print("Visualize sampleId - ", sample)

        sample_index = fixation_sample.index[0]
        scanpath_sample = scanpath_specific_stim_df.loc[sample_index]
        s_stimName = scanpath_sample.stimName
        imgToPlot_size = (self.stim.size[0], self.stim.size[1])

        print('Log..... visualizing fixation map')
        self.map(fixation_sample.fixationMap.values[0], imgToPlot_size, self.stimpath, f_stimName)
        print('Log... visualizing scanpath')
        self.scanpath(scanpath_sample.scanpath, imgToPlot_size, self.stimpath, s_stimName, False)

        return