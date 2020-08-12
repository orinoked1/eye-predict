from modules.data.datasets import DatasetBuilder
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import shapiro
from keras.preprocessing import sequence
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import ks_2samp


datasetbuilder = DatasetBuilder()

#seed = 1010
#stims_array, scanpath_df, fixation_df = datasetbuilder.load_data_pickle()
stimType = "Face"

#for bin_count in [2, 5, 10]:
#bin_count = 5
#scanpath_lan = 3000


fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)
print("p")

def fix_bid_corr_by_subject(fixation_event_data):
    fixation_event_data['x_axis_std'] = fixation_event_data.groupby('sampleId')['avg_X_axis'].transform(np.std)
    fixation_event_data['y_axis_std'] = fixation_event_data.groupby('sampleId')['avg_Y_axis'].transform(np.std)
    fixation_event_data['x_axis_mean'] = fixation_event_data.groupby('sampleId')['avg_X_axis'].transform(np.mean)
    fixation_event_data['y_axis_mean'] = fixation_event_data.groupby('sampleId')['avg_Y_axis'].transform(np.mean)
    fixation_event_data['x_axis_median'] = fixation_event_data.groupby('sampleId')['avg_X_axis'].transform(np.median)
    fixation_event_data['y_axis_median'] = fixation_event_data.groupby('sampleId')['avg_Y_axis'].transform(np.median)
    fixation_event_data['duration_mean'] = fixation_event_data.groupby('sampleId')['duration'].transform(np.mean)
    fixation_event_data['duration_median'] = fixation_event_data.groupby('sampleId')['duration'].transform(np.median)
    fixation_event_data['duration_std'] = fixation_event_data.groupby('sampleId')['duration'].transform(np.std)
    fixation_event_data.drop_duplicates(subset=['sampleId'], inplace=True)
    corr_x_axis_std = fixation_event_data.groupby('subjectID')[['bid', 'x_axis_std']].corr().iloc[0::2, -1]
    corr_y_axis_std = fixation_event_data.groupby('subjectID')[['bid', 'y_axis_std']].corr().iloc[0::2, -1]
    corr_x_axis_mean = fixation_event_data.groupby('subjectID')[['bid', 'x_axis_mean']].corr().iloc[0::2, -1]
    corr_y_axis_mean = fixation_event_data.groupby('subjectID')[['bid', 'y_axis_mean']].corr().iloc[0::2, -1]
    corr_x_axis_median = fixation_event_data.groupby('subjectID')[['bid', 'x_axis_median']].corr().iloc[0::2, -1]
    corr_y_axis_median = fixation_event_data.groupby('subjectID')[['bid', 'y_axis_median']].corr().iloc[0::2, -1]
    corr_duration_mean = fixation_event_data.groupby('subjectID')[['bid', 'duration_mean']].corr().iloc[0::2, -1]
    corr_duration_median = fixation_event_data.groupby('subjectID')[['bid', 'duration_median']].corr().iloc[0::2, -1]
    corr_duration_std = fixation_event_data.groupby('subjectID')[['bid', 'duration_std']].corr().iloc[0::2, -1]


    sList = fixation_event_data.subjectID.unique()
    allCorr= []
    for sId in sList:
        df_by_subject = fixation_event_data[fixation_event_data["subjectID"] == sId]
        df_by_subject['x_axis_std'] = df_by_subject.groupby('sampleId')['avg_X_axis'].transform(np.std)



        all_std = []
        for scanpath in df_by_subject.scanpath.values:
            _, indices = np.unique(scanpath, axis=0, return_index=True)
            uniques_scanpath = scanpath[np.sort(indices)]
            # Normelize
            uniques_scanpath = (uniques_scanpath - uniques_scanpath.min()) / (
                        uniques_scanpath.max() - uniques_scanpath.min())
            bids = np.asanyarray(df_by_subject.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            uniques_scanpath_f = uniques_scanpath.flatten()
            std = np.std(uniques_scanpath_f)
            mean = np.mean(uniques_scanpath_f)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def ranking_distribution(df_by_stim):

    ranks = np.asanyarray(df_by_stim.bid_x.values)
    ranks_mean = np.mean(ranks)
    ranks_median = np.median(ranks)
    ranks_std = np.std(ranks)
    title = "Ranking Distribution "
    xlabel = "Rank"
    ylabel = "Count"

    fig = plt.figure(1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(ranks)
    plt.legend(loc='best')
    plt.savefig(
        "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/statistics/" + title + '.pdf',
        bbox_inches='tight')

    return ranks_mean, ranks_median, ranks_std


def ranking_distribution_per_img(df_by_stim):
    df = df_by_stim.pivot_table(index='stimName', columns='subjectId', values='bid_x')
    gaus_counter = 0
    allKsTestsData = []
    for index in df.iterrows():
        img = index[0]
        print(img)
        title = "ranking_distribution_per_img - " + img
        array1 = np.asanyarray(df.loc[img])
        # remove nan values
        nan_array = np.isnan(array1)
        not_nan_array = ~ nan_array
        array2 = array1[not_nan_array]
        data = array2
        # q-q plot
        qqplot(data, line='s')
        # Hist plot
        plt.hist(array2)
        plt.savefig("/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/statistics/ranking_per_img_plot/" + title + '.pdf', bbox_inches='tight')
        # Shapiro-Wilk Test
        # normality test
        stat, p = shapiro(data)
        # ks-test
        ksTest = []
        for index in df.iterrows():
            img1 = index[0]
            print(img1)
            array3 = np.asanyarray(df.loc[img1])
            # remove nan values
            nan_array = np.isnan(array3)
            not_nan_array = ~ nan_array
            array4 = array3[not_nan_array]
            statistic, pvalue = ks_2samp(array2, array4)
            if pvalue < 0.100:
                ksTest.append(0)
            else:
                ksTest.append(1)
            print('KS Test Statistics=%.3f, p=%.3f' % (statistic, pvalue))
        allKsTestsData.append(ksTest)
        print('Shapiro-Wilk Test Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
            gaus_counter += 1
        else:
            print('Sample does not look Gaussian (reject H0)')
    print('Images with Gaussian ranking distribution:')
    print((gaus_counter / 150) * 100, '%')

    df['avg'] = df.mean(axis=1)
    dfTemp = df.drop('avg', axis=1)
    df['median'] = dfTemp.median(axis=1)
    dfTemp = df.drop(['avg', 'median'], axis=1)
    df['std'] = dfTemp.std(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)


def ranking_dist_per_subject(df_by_stim):
    df = df_by_stim.pivot_table(index='subjectId', columns='stimName', values='bid_x')
    title = "ranking_distribution_per_subject"
    gaus_counter = 0
    allKsTestsData = []
    for index in df.iterrows():
        subject = index[0]
        print(subject)
        title = "ranking_distribution_per_img - " + subject
        array1 = np.asanyarray(df.loc[subject])
        # remove nan values
        nan_array = np.isnan(array1)
        not_nan_array = ~ nan_array
        array2 = array1[not_nan_array]
        data = array2
        # q-q plot
        qqplot(data, line='s')
        # Hist plot
        plt.hist(array2)
        plt.savefig(
            "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/statistics/ranking_per_subject_plot/" + title + '.pdf',
            bbox_inches='tight')
        # Shapiro-Wilk Test
        # normality test
        stat, p = shapiro(data)
        # ks-test
        ksTest = []
        for index in df.iterrows():
            subject1 = index[0]
            print(subject1)
            array3 = np.asanyarray(df.loc[subject1])
            # remove nan values
            nan_array = np.isnan(array3)
            not_nan_array = ~ nan_array
            array4 = array3[not_nan_array]
            statistic, pvalue = ks_2samp(array2, array4)
            if pvalue < 0.100:
                ksTest.append(0)
            else:
                ksTest.append(1)
            print('KS Test Statistics=%.3f, p=%.3f' % (statistic, pvalue))
        allKsTestsData.append(ksTest)
        print('Shapiro-Wilk Test Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
            gaus_counter += 1
        else:
            print('Sample does not look Gaussian (reject H0)')
    print('Images with Gaussian ranking distribution:')
    print((gaus_counter / 52) * 100, '%')

    df['avg'] = df.mean(axis=1)
    dfTemp = df.drop('avg', axis=1)
    df['median'] = dfTemp.median(axis=1)
    dfTemp = df.drop(['avg', 'median'], axis=1)
    df['std'] = dfTemp.std(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)

def unique_flatten_scanpath_bid_corr_by_subject(df_by_stim):
    sList = df_by_stim.subjectId.unique()
    allCorr= []
    for sId in sList:
        df_by_subject = df_by_stim[df_by_stim["subjectId"] == sId]
        all_std = []
        for scanpath in df_by_subject.scanpath.values:
            _, indices = np.unique(scanpath, axis=0, return_index=True)
            uniques_scanpath = scanpath[np.sort(indices)]
            # Normelize
            uniques_scanpath = (uniques_scanpath - uniques_scanpath.min()) / (
                        uniques_scanpath.max() - uniques_scanpath.min())
            bids = np.asanyarray(df_by_subject.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            uniques_scanpath_f = uniques_scanpath.flatten()
            std = np.std(uniques_scanpath_f)
            mean = np.mean(uniques_scanpath_f)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def flatten_scanpath_bid_corr_by_subject(df_by_stim):
    sList = df_by_stim.subjectId.unique()
    allCorr= []
    for sId in sList:
        df_by_subject = df_by_stim[df_by_stim["subjectId"] == sId]
        all_std = []
        for scanpath in df_by_subject.scanpath.values:
            # Normelize
            scanpath = (scanpath - scanpath.min()) / (
                    scanpath.max() - scanpath.min())
            bids = np.asanyarray(df_by_subject.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            uniques_scanpath_f = scanpath.flatten()
            std = np.std(uniques_scanpath_f)
            mean = np.mean(uniques_scanpath_f)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def x_scanpath_bid_corr_by_subject(df_by_stim):
    sList = df_by_stim.subjectId.unique()
    allCorr= []
    for sId in sList:
        df_by_subject = df_by_stim[df_by_stim["subjectId"] == sId]
        all_std = []
        for scanpath in df_by_subject.scanpath.values:
            # Normelize
            scanpath = (scanpath - scanpath.min()) / (
                    scanpath.max() - scanpath.min())
            bids = np.asanyarray(df_by_subject.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            x_coordinate = scanpath[:, 0]
            std = np.std(x_coordinate)
            mean = np.mean(x_coordinate)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def y_scanpath_bid_corr_by_subject(df_by_stim):
    sList = df_by_stim.subjectId.unique()
    allCorr= []
    for sId in sList:
        df_by_subject = df_by_stim[df_by_stim["subjectId"] == sId]
        all_std = []
        for scanpath in df_by_subject.scanpath.values:
            # Normelize
            scanpath = (scanpath - scanpath.min()) / (
                    scanpath.max() - scanpath.min())
            bids = np.asanyarray(df_by_subject.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            y_coordinate = scanpath[:, 1]
            std = np.std(y_coordinate)
            mean = np.mean(y_coordinate)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def x_scanpath_bid_corr_by_img(df_by_stim):
    imgList = df_by_stim.stimName.unique()
    allCorr= []
    for img in imgList:
        df_by_img = df_by_stim[df_by_stim["stimName"] == img]
        all_std = []
        for scanpath in df_by_img.scanpath.values:
            # Normelize
            scanpath = (scanpath - scanpath.min()) / (
                    scanpath.max() - scanpath.min())
            bids = np.asanyarray(df_by_img.bid_x.values)
            bids = (bids - bids.min()) / (
                    bids.max() - bids.min())
            x_coordinate = scanpath[:, 0]
            std = np.std(x_coordinate)
            mean = np.mean(x_coordinate)
            all_std.append(std)
        all_std = np.asanyarray(all_std)
        corr = np.corrcoef(all_std, bids)
        allCorr.append(corr[0][1])
    return np.asanyarray(allCorr)

def scanpath_distributions_by_img(df_by_stim):
    imgList = df_by_stim.stimName.unique()
    allKsTestsData = []
    for img in imgList:
        df_by_subject = df_by_stim[df_by_stim["stimName"] == img]
        allDistDiscrip = []
        for scanpath, sId in zip(df_by_subject.scanpath.values, df_by_subject.subjectId.values):
            title = "scanpath dist, subject - " + img + ", " + sId
            _, indices = np.unique(scanpath, axis=0, return_index=True)
            uniques_scanpath = scanpath[np.sort(indices)]
            # Normelize
            uniques_scanpath = (uniques_scanpath - uniques_scanpath.min()) / (uniques_scanpath.max() - uniques_scanpath.min())
            uniques_scanpath_f = uniques_scanpath.flatten()
            # Hist plot
            plt.hist(uniques_scanpath_f)
            plt.savefig(
                "/export/home/DATA/schonberglab/pycharm_eyePredict/etp_data/processed/figs/statistics/scanpath_plots_per_img/" + title + '.pdf',
                bbox_inches='tight')
            # ks-test
            ksTest = []
            for scanpath1 in df_by_subject.scanpath.values:
                scanpathList = []
                _, indices = np.unique(scanpath1, axis=0, return_index=True)
                uniques_scanpath1 = scanpath1[np.sort(indices)]
                # Normelize
                uniques_scanpath1 = (uniques_scanpath1 - uniques_scanpath1.min()) / (
                        uniques_scanpath1.max() - uniques_scanpath1.min())
                uniques_scanpath1_f = uniques_scanpath1.flatten()
                scanpathList.append(uniques_scanpath_f)
                scanpathList.append(uniques_scanpath1_f)
                np.asanyarray(scanpathList)
                scanpathList = sequence.pad_sequences(scanpathList, dtype='float32', maxlen=800)
                statistic, pvalue = ks_2samp(scanpathList[0], scanpathList[1])
                if pvalue < 0.100:
                    ksTest.append(0)
                else:
                    ksTest.append(1)
                print('KS Test Statistics=%.3f, p=%.3f' % (statistic, pvalue))
            allKsTestsData.append(ksTest)


fix_bid_corr_by_subject(fixation_event_data)

corr1 = unique_flatten_scanpath_bid_corr_by_subject(df_by_stim)
corr2 = flatten_scanpath_bid_corr_by_subject(df_by_stim)
corr3 = x_scanpath_bid_corr_by_subject(df_by_stim)
corr4 = y_scanpath_bid_corr_by_subject(df_by_stim)
corr5 = x_scanpath_bid_corr_by_img(df_by_stim)

median5 = np.median(corr5)
median2 = np.median(corr2)
median3 = np.median(corr3)
per3 = np.percentile(corr3, 70)
per33 = np.percentile(corr3, 90)
per333 = np.percentile(corr3, 30)
median4 = np.median(corr4)

df_by_subject = df_by_stim[df_by_stim["subjectId"] == '106']


def slope(x1, y1, x2, y2):
    if x2 == x1:
        m = None
    else:
        m = (y2-y1)/(x2-x1)
    return m

def euclidian_distance(x1, y1, z1, x2, y2, z2):
    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2) + np.square(z1 - z2))
    return dist


