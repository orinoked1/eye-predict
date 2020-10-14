import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

stimType = 'Face'
datasetbuilder = DatasetBuilder()
fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)

face_fixations_scanpath_df = datasetbuilder.get_fixations_scanpath_df(fixation_event_data)
# remove small scanpath samples
face_fixations_scanpath_df['scanpath_len'] = 0
for i in range(face_fixations_scanpath_df.scanpath.size):
    face_fixations_scanpath_df.at[i, 'scanpath_len'] = len(face_fixations_scanpath_df.scanpath[i])
face_fixations_scanpath_df = face_fixations_scanpath_df[face_fixations_scanpath_df['scanpath_len'] > 5]
by_x_df = face_fixations_scanpath_df

lda_mean_scores = []
linearSvm_mean_scores = []
radialSvm_mean_scores = []
# run by image
#for stim in face_fixations_scanpath_df.stimName.unique():
#    print(stim)
#    by_x_df = face_fixations_scanpath_df[face_fixations_scanpath_df['stimName'] == stim]

# run by subject
#for subject in face_fixations_scanpath_df.subjectID.unique():
#    print(subject)
#    by_x_df = face_fixations_scanpath_df[face_fixations_scanpath_df['subjectID'] == subject]

# add five bins for ranking
by_x_df['five_bin_bid'] = pd.qcut(by_x_df.bid, 5, labels=[0, 1, 2, 3, 4])

### Compute one HMM per scanpath

num_clusters = 3
def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [round(f/s, 4) for f in row]
    return M

feature_vectors = []
# get HMM features flattened and normalized
for scanpath in by_x_df.scanpath:
    # run only on x,y coordinates
    twodScanpath = scanpath[:, [1, 2]]
    # calculate 3 clusters using k-means for prticipent ROIs
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(twodScanpath)
    # clusters mean as the states centers
    clusters_mean = kmeans.cluster_centers_.flatten()
    #normalize states centers
    clusters_mean -= np.mean(clusters_mean)
    if np.std(clusters_mean) == 0:
        clusters_mean /= 1
    else:
        clusters_mean /= np.std(clusters_mean)
    # get the states sequence
    sequence = kmeans.predict(twodScanpath)
    # create transition matrix for the sequence
    m = transition_matrix(sequence)
    m = np.asanyarray(m).flatten()
    # normalize states centers
    m -= np.mean(m)
    if np.std(m) == 0:
        m /= 1
    else:
        m /= np.std(m)
    # calculate states priors
    unique_elements, counts_elements = np.unique(sequence, return_counts=True)
    n = sequence.size
    prior = np.around(counts_elements/n, decimals=4).flatten()
    # normalize states centers
    prior -= np.mean(prior)
    if np.std(prior) == 0:
        prior /= 1
    else:
        prior /= np.std(prior)
    # calculate states variance
    seq = sequence.reshape(-1, 1)
    np_concat = np.concatenate((twodScanpath, seq), axis=1)
    df = pd.DataFrame(data=np_concat, columns=["x", "y", "state"])
    variance = df.groupby(['state']).var(ddof=0)
    variance = np.asanyarray(variance).flatten()
    # normalize states centers
    type = variance.dtype.str
    if type != '<f8':
        variance = variance.astype(np.float64)
    variance -= np.mean(variance)
    if np.std(prior) == 0:
        variance /= 1
    else:
        variance /= np.std(variance)
    # build one feature vector for all features
    features = np.concatenate((prior, m, clusters_mean, variance), axis=0)
    feature_vectors.append(features)



# run LDA and SVM for classification
x = np.asanyarray(feature_vectors)
y = by_x_df[['five_bin_bid']].values
# split train test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
# LDA
#clf = LinearDiscriminantAnalysis()
#clf.fit(x_train, y_train)
#scoreClf = clf.score(x_test, y_test)
#lda_mean_scores.append(scoreClf)
# linear SVC
clf = SVC(kernel='linear', gamma='auto')
svmFit = clf.fit(x_train, y_train)
scoreLinearSvm = clf.score(x_test, y_test)
linearSvm_mean_scores.append(scoreLinearSvm)
# radial SVC
#clf = SVC(kernel='rbf', gamma='auto')
#svmFit = clf.fit(x_train, y_train)
#scoreRadialSvm = clf.score(x_test, y_test)
#radialSvm_mean_scores.append(scoreRadialSvm)

# SVR
x = np.asanyarray(feature_vectors)
y = by_x_df[['bid']].values
# split train test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
clf = SVR(kernel='linear', gamma='auto')
svmFit = clf.fit(x_train, y_train)
scoreLinearSvr = clf.score(x_test, y_test)


print('LDA mean score: ')
print(np.mean(lda_mean_scores))
print('linear SVM mean score: ')
print(np.mean(linearSvm_mean_scores))
print('Radial SVM mean score: ')
print(np.mean(radialSvm_mean_scores))
print('stop')

    # SVM for regression
    #x = np.asanyarray(feature_vectors)
    #y = by_stim_df[['bid']].values
    # split train test datasets
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # linear SVR
    #regr = SVR(kernel='linear', gamma='auto')
    #svmFit = regr.fit(x_train, y_train)
    #scoreLinearSvr = regr.score(x_test, y_test)
    # radial SVR
    #regr = SVR(kernel='rbf', gamma='auto')
    #svmFit = regr.fit(x_train, y_train)
    #scoreRadialSvr = regr.score(x_test, y_test)
