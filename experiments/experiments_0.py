#%%
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance

from tqdm import tqdm

import random
import time
import pickle

import sys
import json
import time

#%%
# drift detection methods
def d3(X,clf = LogisticRegression(solver='liblinear')):
    y = np.ones(X.shape[0])
    y[:int(X.shape[0]/2)] = 0   # set the fist window to class 0, the second one to class 1
    
    predictions = np.zeros(y.shape)
    
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs   # at each fold, a subset of predictions is filled in
    auc_score = roc(y, predictions)   # if AUC is > 0.5, then there is drift!
    
    return 1 - auc_score

# drift localization / explanation methods
def FeatureImportanceRF(X,y, **params):   # forest with 100 trees
    return FeatureImportance0(X,y,  RandomForestRegressor(), **params)

def FeatureImportanceET(X,y, **params):
    return FeatureImportance0(X,y,ExtraTreesRegressor(),**params)

def FeatureImportance0(X,y, model, **params):
    model = model.fit(X,y)
    return model, model.feature_importances_

def PermutationFeatureImportanceRF(X,y, **params):
    return PermutationFeatureImportance0(X,y,RandomForestRegressor())

def PermutationFeatureImportanceET(X,y, **params):
    return PermutationFeatureImportance0(X,y,ExtraTreesRegressor(), **params)

def PermutationFeatureImportance0(X,y, model, **params):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train,y_train)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=1)
    return model, result.importances_mean



#%%
# read baseline file
df_baseline = pd.read_csv('Baseline.csv')
# remove timestamp column
df_baseline = df_baseline.drop(columns=['Unnamed: 0'])

# read measurements file
# df = pd.read_csv('Measurements.csv', delimiter=';', decimal=',')   # p257, incipient, 11 mm
df = pd.read_csv('Pressures_p461.csv') 
# remove timestamp column
df = df.drop(columns=['Timestamp'])
sensor_cols = df.columns.tolist()
# downsample to 1 every third row
df = df.iloc[::3, :].reset_index(drop=True)

hour_len = int(60/15)
day_len = 24*hour_len
week_len = 7*day_len
month_len = 4*week_len

# %%
# generate data
# leak_start = 90 * day_len

# concatenate baseline and leak data at leak start as numpy array
# data = pd.concat([df_baseline.iloc[:leak_start], df.iloc[leak_start:]], axis=0).reset_index(drop=True).to_numpy()
data = df.to_numpy()

#%%
# 1. DRIFT DETECTION
# run drift detection on sliding windows (one check every day, windows of two weeks)
i = 0
results = []
while i + 2*week_len < data.shape[0]:
    X = data[i:i+2*week_len, :]
    result = d3(X)
    results.append(result)
    i += day_len

print("The drift detection minimum score is obtained at day:", np.argmin(results))

# plot results
import matplotlib.pyplot as plt
plt.plot(results)
plt.xlabel('Days (processed every day)')
plt.ylabel('Drift Detection Score')
plt.title('Drift Detection Over Time')
plt.show()

# %%
# 2. DRIFT LOCALIZATION / EXPLANATION
# select the window with minimum score
i = np.argmin(results) * day_len
X = data[i:i+2*week_len, :]

methods = {fun.__name__:fun for fun in 
    [FeatureImportanceET, FeatureImportanceRF, PermutationFeatureImportanceRF,PermutationFeatureImportanceET]}


def drift_explanation(X, method, split, **params):
    y = np.array(X.shape[0]//2*[0] + X.shape[0]//2*[1])  # first half is class 0, second half is class 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42*split)
    
    t0 = time.time()
    model,feature_scores = methods[method](X_train,y_train, **params)
    t1 = time.time()

    if len(feature_scores.shape) != 1:
        print("Deformed feature scores: '%s' %s"%(method,str(feature_scores.shape)))
        feature_scores = feature_scores.ravel()
    
    return {"feature_scores":json.dumps(dict(zip(sensor_cols,map(lambda x: "%.5f"%x, list(feature_scores))))), "model_score": model.score(X_test,y_test), "time":t1-t0}


explanation_result = drift_explanation(X, method='FeatureImportanceRF', split=1)
feature_scores = json.loads(explanation_result['feature_scores'])

# print pair with higher feature score
max_feature = max(feature_scores, key=feature_scores.get)
print("Highest feature score:", max_feature, feature_scores[max_feature])

# %%
