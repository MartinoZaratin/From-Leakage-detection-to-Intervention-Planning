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
import seaborn as sns

from tqdm import tqdm
import json
import random
import time

import glob
import os



#%%
# -------------------
# define drift detection methods
#-------------------
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

def knn_d3(X, n_neighbors=200):
    return d3(X, clf=KNeighborsClassifier(n_neighbors=n_neighbors))

# def knn_d3_20(X):
#     return knn_d3(X, n_neighbors=20)

# def knn_d3_50(X):
#     return knn_d3(X, n_neighbors=50)

# def knn_d3_100(X):
#     return knn_d3(X, n_neighbors=100)


def ks(X, s=None):
    if s is None:
        s = int(X.shape[0]/2)

    D_values = [ks_2samp(X[:s, i], X[s:, i], mode="exact")[0] for i in range(X.shape[1])]   # compute KS D statistic for each sensor node
    scores = [1 - D for D in D_values]

    return min(scores)




# ----------------------
# experiment function
#-----------------------
def run_dd_experiments(file, leak_node_1, leak_node_2, leak_diam_mm_1, leak_diam_mm_2, t_start_1, t_start_2, closest_sensor_info, df_baseline, dd_methods, dd_params):
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

    # 1. DRIFT DETECTION
    # run drift detection on sliding windows (one check every day, windows of two weeks)
    for j, dd_method in enumerate(dd_methods):
        i = 0
        dd_results = []
        while i + 2*week_len < data.shape[0]:
            X = data[i:i+2*week_len, :]
            dd_result = dd_method(X)
            dd_results.append((dd_result, (i + 2* day_len <= t_start_1) and (t_start_2 < i - 2* day_len + 2*week_len)))  # store also if drift is in the window (2 days margin)
            i += day_len

        # compute roc AUC for drift detection results
        y_true = [1 if r[1] else 0 for r in dd_results]
        y_scores = [r[0] < dd_params[j] for r in dd_results]
        auc_score = roc(y_true, y_scores)

        record[f"dd_auc_{dd_method.__name__}"] = auc_score

        drift_timepoint = (np.argmin([r[0] for r in dd_results]) + 7) * day_len * 15 * 60  # in seconds
                                                                            # add 7 days to center the window on the detection point

        # store results
        record[f"dd_timepoint_{dd_method.__name__}"] = drift_timepoint
        record[f"dd_score_{dd_method.__name__}"] = min([r[0] for r in dd_results])


    # optional: visualize results
    plots = False
    if plots:
        # use Seaborn styling for Matplotlib
        sns.set_theme(style="whitegrid", context="talk")

        x_days = np.arange(len(dd_results))
        y_auc = [1 - r[0] for r in dd_results]
        threshold = 1 - dd_params[0]
    
        plt.figure(figsize=(12, 7))
        # Base line (below threshold)
        plt.plot(x_days, y_auc, color='dimgray', linewidth=2, alpha=0.5, label='D3 AUC score')
        # Overlay the bold section above the threshold
        plt.plot(np.array(x_days)[np.array(y_auc) > threshold], np.array(y_auc)[np.array(y_auc) > threshold],
            color='navy', linewidth=4, label='Flag drift')
        # Add threshold
        plt.axhline(y=1 - dd_params[0], color="navy", linestyle="--", linewidth=2, label="Drift detection threshold")
        # Add vertical lines
        # plt.axvline(x=(split_point - 2 * week_len) / day_len, color="green", linestyle="--", linewidth=2, label="Leak start")
        # plt.axvline(x=(split_point - week_len) / day_len, color="black", linestyle=":", linewidth=2, label="Peak samples dissimilarity")
        # # Highlight drift window
        # plt.axvspan((split_point - 2 * week_len) / day_len, (split_point) / day_len, color="green", alpha=0.1, label="Time window with drift")
        # Labels and title
        plt.xlabel("Time (days)", fontsize=14)
        plt.ylabel("AUC score", fontsize=14)
        # Legend and grid
        plt.legend(fontsize=12, fancybox=True, loc="upper right")
        plt.grid(True, which="major", linestyle="--", alpha=0.6)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0.4, 1)
        plt.tight_layout()
        save_figs = False
        # if save_figs:
            # plt.savefig(f"drift_detection_leak_{leak_node}_diam_{leak_diam_mm}.png", dpi=300)
        plt.show()


    # append record to results
    results.append(record)

    results_df = pd.DataFrame(results)

    return results_df




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
# with open("top2_shortest_paths_to_sensors.json", "r") as f:
#     closest_sensor_info = json.load(f)
closest_sensor_info = {}

# read baseline file
df_baseline = pd.read_csv('..\\Baseline.csv')
# remove timestamp column
df_baseline = df_baseline.drop(columns=['Unnamed: 0'])

n_repeats = 1   # number of repetitions of the experiment for each leaking pipe



dd_methods = [d3, ks]   # drift detection methods to evaluate

# parameters for drift detection methods
# d3 --> 0.4
# ks --> 0.825

# run experiments

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
    t_start_2 = t_start_1 + random.randint(0, week_len)  # second leak starts within a week of the first
    
    # Run your existing experiment
    results_df = run_dd_experiments(
        file,
        leak_node_1,
        leak_node_2,
        leak_diam_mm_1,
        leak_diam_mm_2,
        t_start_1,
        t_start_2,
        closest_sensor_info,
        df_baseline,
        dd_methods,
        dd_params=[0.4, 0.825]  # example thresholds for d3, knn_d3, ks
    )
    
    all_results.append(results_df)

# Concatenate all results into one DataFrame
results_all = pd.concat(all_results, ignore_index=True)

# save results to CSV
save_results = True
if save_results:
    results_all.to_csv("dd_results_two_leaks.csv", index=False)

# %%
