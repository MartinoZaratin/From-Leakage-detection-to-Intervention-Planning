#%%
# --------------------
# Load libraries
# --------------------
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

import matplotlib.pyplot as plt

from tqdm import tqdm
import json
import random
import time
import glob
import os

#%%
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

# ----------------------
# experiment function
#-----------------------
def run_explanation_experiments(file, leak_node_1, leak_node_2, leak_diam_mm_1, leak_diam_mm_2,
                            t_start_1, t_start_2,
                            closest_sensor_info, df_baseline,
                            explanation_methods, explanation_params):
    results = []

    # read file with one leak
    df_1 = pd.read_csv(f'..\\..\\dataset generator\\Datasets\\Leakages {leak_diam_mm_1}mm\\leakage_{leak_node_1}_{leak_diam_mm_1}.csv')
    # remove timestamp column
    df_1 = df_1.drop(columns=['Unnamed: 0'])

    # read file with both leaks
    df_2 = pd.read_csv(file)
    df_2 = df_2.drop(columns=['Unnamed: 0'])

    print(f"Running experiment for leak node {leak_node_1} and {leak_node_2}...")
    print(f"Start times: leak 1 at {t_start_1/day_len} days, leak 2 at {t_start_2/day_len} days")


    # initialize record
    record = {"leak_node_1": leak_node_1,
            "leak_node_2": leak_node_2,
            "leak_diam_mm_1": leak_diam_mm_1,
            "leak_diam_mm_2": leak_diam_mm_2,
            "leak_1_at": t_start_1 * 15 * 60,     # in seconds
            "leak_2_at": t_start_2 * 15 * 60,     # in seconds
            "time_displacement": (t_start_2 - t_start_1) * 15 * 60   # in seconds
            }

    # concatenate baseline and leak data at split point as numpy array
    data = pd.concat([df_baseline.iloc[:t_start_1], df_1.iloc[t_start_1:t_start_2], df_2.iloc[t_start_2:]], axis=0).reset_index(drop=True).to_numpy()
    
    # create detection timepoint by adding displacement to the average of t_start_1 and t_start_2
    split_point = (t_start_1 + t_start_2) // 2
    # displacement is randomly samples in the range [-24h, +24h]
    displacement = random.randint(-96, 96)  # in time steps (15 min each)
    record["detection_timepoint (s)"] = (split_point + displacement) * 15 * 60  # in seconds
    detection_timepoint = split_point + displacement

    # DRIFT LOCALIZATION / EXPLANATION

    # train - test split for explanation
    # adjust window if displacement causes out-of-bounds
    if detection_timepoint - week_len < 0:
        detection_timepoint = week_len
    if detection_timepoint + week_len > data.shape[0]:
        detection_timepoint = data.shape[0] - week_len
    X = data[detection_timepoint - week_len : detection_timepoint + week_len, :]
    y = np.array(X.shape[0]//2*[0] + X.shape[0]//2*[1])  # first half is class 0, second half is class 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    for i , explanation_method in enumerate(explanation_methods):
        # run explanations
        model, feature_scores = explanation_method(X_train, y_train, **explanation_params[i])

        # ensure correct shape
        if len(feature_scores.shape) != 1:
            print("Deformed feature scores: '%s' %s"%(explanation_method,str(feature_scores.shape)))
            feature_scores = feature_scores.ravel()

        record[f"feature_scores_{explanation_method.__name__}"] = {col: f"{score:.5f}" for col, score in zip(df_baseline.columns, feature_scores)}
        record[f"explanation_model_score_{explanation_method.__name__}"] = model.score(X_test, y_test)
        record[f"best_explained_sensor_{explanation_method.__name__}"] = df_baseline.columns[np.argmax(feature_scores)]

    # append record to results
    results.append(record)

    results_df = pd.DataFrame(results)

    return results_df

#%%
# -------------------
# Setup experiment
#--------------------
# set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# define helper variables for time windows
hour_len = int(60/15)
day_len = 24*hour_len
week_len = 7*day_len
month_len = 4*week_len

# read json file with sensor proximity information
with open("..\\top2_shortest_paths_to_sensors.json", "r") as f:
    closest_sensor_info = json.load(f)

# read baseline file
df_baseline = pd.read_csv('..\\Baseline.csv')
# remove timestamp column
df_baseline = df_baseline.drop(columns=['Unnamed: 0'])

n_repeats = 1   # number of repetitions of the experiment for each leaking pipe


#%%
#---------------------
# Run experiments
#--------------------


# define explanations
explanation_methods = [PermutationFeatureImportanceET]

# List to store results for each leak size
all_results = []

folder = "..\\..\\dataset generator\\Datasets\\Multiple leaks"
csv_files = glob.glob(os.path.join(folder, "*.csv"))  # list all CSVs

for file in csv_files:
    # extract leak nodes and diameters from filename
    filename = os.path.basename(file)
    parts = filename.replace("leakage_", "").replace(".csv", "").split("_")
    leak_node_1 = parts[0]
    leak_diam_mm_1 = int(parts[1])
    leak_node_2 = parts[2]
    leak_diam_mm_2 = int(parts[3])

    # set leak starting point
    t_start_1 = random.randint(week_len, 3* month_len - 2* week_len - 1)
    t_start_2 = t_start_1 + random.randint(0, 2*day_len)  # second leak starts within two days of the first
    
    # Run your existing experiment
    results_df = run_explanation_experiments(
        file,
        leak_node_1,
        leak_node_2,
        leak_diam_mm_1,
        leak_diam_mm_2,
        t_start_1,
        t_start_2,
        closest_sensor_info,
        df_baseline,
        explanation_methods,
        explanation_params=[{} for _ in explanation_methods]
    )
    
    all_results.append(results_df)

# Concatenate all results into one DataFrame
results_all = pd.concat(all_results, ignore_index=True)

# save results to CSV
save_results = True
if save_results:
    results_all.to_csv("explanation_results_two_leaks.csv", index=False)
