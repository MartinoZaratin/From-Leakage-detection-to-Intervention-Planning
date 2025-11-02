#%%
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from tqdm import tqdm

import random
import time
import pickle

import sys
import json
import time

#%%
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
        predictions[test_idx] = probs
    auc_score = roc(y, predictions)   # if AUC is > 0.5, then there is drift!
    
    return 1 - auc_score

#%%
# read measurements excel file
# df = pd.read_excel('Measurements.xlsx')   # a leak is present at Jan 8
df_base = pd.read_csv('Baseline.csv')
df_base = df_base.drop(columns=['Unnamed: 0'])

# remove timestamp column
# df_base = df_base.drop(columns=['Timestamp'])

# downsample to every 3rd row
# df = df.iloc[::3, :].reset_index(drop=True)

df_leak = pd.read_csv('leakage_n10_15.csv')
df_leak = df_leak.drop(columns=['Unnamed: 0'])

# df is made of the first half of base and second half of leak
df = pd.concat([df_base.iloc[:len(df_base)//2], df_leak.iloc[len(df_leak)//2:]], axis=0)
df = df.reset_index(drop=True)


hour_len = int(60/15)
day_len = 24*hour_len
week_len = 7*day_len
month_len = 4*week_len

#%%
# select first two weeks of measurements
# X = df.iloc[:2*week_len, :].values
# result = d3(X)
# print(result)

# %%
# sliding window approach
i = 0
results = []
while i + 2*week_len < df.shape[0]:
    if not i % month_len:
        print(f"Processing month {i//month_len + 1}")
    # process once a day
    if not i % day_len:
        X = df.iloc[i:i+2*week_len, :].values
        result = d3(X)
        results.append(result)
        i += day_len
    else:
        i += 1
        print('Ooops')


# %%
# plot results
import matplotlib.pyplot as plt
plt.plot(results)
plt.xlabel('Days (processed every day)')
plt.ylabel('Drift Detection Score')
plt.title('Drift Detection Over Time')
plt.show()

# %%
# plot data
data = pd.read_csv('Measurements.csv', delimiter=';', decimal=',')
data = data.drop(columns=['Timestamp'])
plt.plot(df)
plt.xlabel('Time Steps (15 min intervals)')
plt.ylabel('Pressure Readings')
plt.title('Pressure Readings Over Time')
plt.show()

# %%
