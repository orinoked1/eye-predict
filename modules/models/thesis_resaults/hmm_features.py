import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from modules.data.datasets import DatasetBuilder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import os
import seaborn as sns


stimType = 'Snack'

datasetbuilder = DatasetBuilder()
fixation_event_data, saccad_event_data = datasetbuilder.load_data_for_fix_sacc_statistics(stimType)

df = datasetbuilder.get_fixations_scanpath_df(fixation_event_data, stimType)
# remove small scanpath samples
df['scanpath_len'] = 0
for i in range(df.scanpath.size):
    df.at[i, 'scanpath_len'] = len(df.scanpath[i])
df = df[df['scanpath_len'] > 5]

# add the visual feature - ranking mean
stim_avg = df.groupby(['stimName'])['bid'].mean().reset_index()
stim_avg.rename(columns={'bid': 'avg'}, inplace=True)

df = df.merge(stim_avg, how='left', on='stimName')
avg_list = df['avg'].tolist()
avg_list = np.array(avg_list).reshape(-1, 1)

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
for scanpath in df.scanpath:
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
    df_variance = pd.DataFrame(data=np_concat, columns=["x", "y", "state"])
    variance = df_variance.groupby(['state']).var(ddof=0)
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

# dataframe for correlation heatmap
data = pd.DataFrame(feature_vectors)

# x y split
x = np.asanyarray(feature_vectors)
# add the ranking mean as a feature to the features list
#x = np.concatenate((x, avg_list), axis=1)

y = df[['bid']].values

# linear regression
linearModel = linear_model.LinearRegression()
fit = linearModel.fit(x, y)
score = linearModel.score(x, y)
predictions = linearModel.predict(x)
rmse = np.sqrt(np.square(np.subtract(y, predictions)).mean())

# Ordinary Least Squares
model = sm.OLS(y, x)
results = model.fit()
print(results.summary())


# calculate features correlations and make a seaborn heatmap
corr = data.corr()
fig = plt.figure(2)
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')

currpath = os.getcwd()
fig.savefig(currpath + "/snack_hmm_features_heat_map.pdf", bbox_inches='tight')



print('x')