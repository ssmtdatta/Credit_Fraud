import os
import pandas as pd
import numpy as np 
import args

from imblearn.under_sampling import NearMiss

import config as _config


'''
sampling
outlier
modeling - tree, rf, naive bayes, svm
feature importance
'''

def apply_nearMiss(X, y, near_miss_type=3):

	'''
	Using NearMiss algorithm, undersamples the majority class to match the 
	sample size of the minority class
	
	Args:
	X (np.array): features associated with examples of majority and minority classes;
	              shape = number of examples x number of features
	y (np.array): lables of each example; 
	              shape = number of examples x 1
	near_miss_type (int): Type of NearMiss algorithm

	Returns:
	X_samp (np.array): features associated with undersampled majority examples and 
	                   all minority class examples; 
	                   shape = 2*number of minority examples x number of features
	y_samp (np.array): lables of each example; 
	                   shape = 2*number of minority examples x 1
	              
	'''
	undersample = NearMiss(version=1, n_neighbors=near_miss_type)
	X_smap, y_samp = undersample.fit_resample(X, y)
	return X_samp, y_samp

	


def apply_kmeans(X, k, randomstate=None):
    model = KMeans(n_clusters=k, random_state=None)
    cluster_labels = model.fit_predict(X)
    cluster_centers = model.cluster_centers_
    inertia = model.inertia_
    return cluster_centers, cluster_labels, inertia
	


print("Read data ...")
path = os.path.join(_config.DATA_DIR, 'ETL')
filename = 'creditcard.csv'

df_input = pd.read_csv( os.path.join(path, filename)).drop_duplicates()
df_input_0 = df_input[ df_input['Class']==0 ]
df_input_1 = df_input[ df_input['Class']==1 ]

target = 'Class'
features = df_input.columns.tolist()
features.remove(target)
features.remove('Time')


# =============
# undersampling
X, y = df_input[features].values, df_input[[target]].values
X_samp, y_samp = apply_nearMiss(X, y, near_miss_type=3)
df_X_samp = pd.DataFrame(X_samp, columns=features)
df_y_samp = pd.DataFrame(y_samp, columns=['Class'])
df_samp = pd.concat([df_X, df_y], axis=1)


# =============
# clustering

k = 100

X0, y0 = df_input_0[features].values, df_input_0[[target]].values
X1, y1 = df_input_1[features].values, df_input_1[[target]].values

cluster_centers, cluster_labels, inertia = apply_kmeans(X0, k)

df_X0_clust = pd.DataFrame(cluster_centers, columns=features)
df_y0_clust = pd.DataFrame(y0, columns=['Class'])
df_0_clust = pd.concat([df_X0_clust df_y0_clust], axis=1)

df_X1_clust = pd.DataFrame(cluster_centers, columns=features)
df_y0_clust = pd.DataFrame(y0, columns=['Class'])
df_1_clust = pd.concat([df_X1_clust, df_y1_clust], axis=1)

df_clust = pd.concat([df_0_clust, df_1_clust], axis=0)

#=============
# combine undersample and cluster output

df = pd.concat([df, df_clust], axis=0)







