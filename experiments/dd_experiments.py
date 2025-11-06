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
def run_dd_experiments(leak_nodes, leak_diam_mm, n_repeats, closest_sensor_info, df_baseline, dd_methods, dd_params):
    results = []

    for leak_node in leak_nodes:
        # read file with leakage data
        df = pd.read_csv(f'..\\dataset generator\\Datasets\\Leakages {leak_diam_mm}mm\\leakage_{leak_node}_{leak_diam_mm}.csv')
        # remove timestamp column
        df = df.drop(columns=['Unnamed: 0'])

        # generate n_repeats random split points for the experiment repetitions
        n_timesteps = df.shape[0]
        split_points = random.sample(range(week_len, n_timesteps - week_len), k=n_repeats)

        # iterate over repetitions
        for split_idx, split_point in enumerate(split_points):
            print(f"Running experiment for leak node {leak_node}, repetition {split_idx+1}/{n_repeats}...")

            # initialize record
            record = {"leak_node": leak_node,
                    "leak_diam_mm": leak_diam_mm,
                    "leak_at": split_point * 15 * 60,     # in seconds
                    "closest_sensor": closest_sensor_info[leak_node]["closest_sensors"][0]}

            # concatenate baseline and leak data at split point as numpy array
            data = pd.concat([df_baseline.iloc[:split_point], df.iloc[split_point:]], axis=0).reset_index(drop=True).to_numpy()

            # 1. DRIFT DETECTION
            # run drift detection on sliding windows (one check every day, windows of two weeks)
            for j, dd_method in enumerate(dd_methods):
                i = 0
                dd_results = []
                while i + 2*week_len < data.shape[0]:
                    X = data[i:i+2*week_len, :]
                    dd_result = dd_method(X)
                    dd_results.append((dd_result, i + 2* day_len <= split_point < i - 2* day_len + 2*week_len))  # store also if drift is in the window (1 day margin)
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


            # optional: visualize results above acertian threshold
            # before running this, the D3 function was changed to return AUC score directly
            if True:
                # # plot dd results
                # plt.figure(figsize=(10,6))
                # # plt.plot(np.arange(len([r[0] for r in dd_results]))*day_len/ hour_len, [r[0] for r in dd_results], marker='o')   # in hours
                # plt.plot(np.arange(len([r[0] for r in dd_results])), [1 - r[0] for r in dd_results], label = 'D3 AUC Score')   # in days
                # # vertical line for the leak start
                # plt.axvline(x=(split_point - 2 * week_len) / day_len, color='r', linestyle='--', label='Leak Start')
                # #vertical line for maximum difference in distribution
                # plt.axvline(x=(split_point - week_len) / day_len, color='b', linestyle='--', label='Maximum separation between the two samples')
                # # horizontal line for the threshold
                # plt.axhline(y=1 - dd_params[0], color='g', linestyle='--', label='Drift Detection Threshold')
                # # highlight week before and after leak
                # plt.axvspan((split_point - 2 * week_len) / day_len, (split_point) / day_len, color='y', alpha=0.3, label='Time window with drift')
                # plt.xlabel('Time (days)')
                # plt.ylim(0.4, 1)
                # plt.title(f'Drift Detection Results for Leak Node {leak_node}, Leak Diameter {leak_diam_mm} mm')
                # plt.legend()
                # plt.show()


                # Optional: use Seaborn styling for Matplotlib
                sns.set_theme(style="whitegrid", context="talk")

                # Prepare data
                x_days = np.arange(len(dd_results))
                y_auc = [1 - r[0] for r in dd_results]
                threshold = 1 - dd_params[0]
                

                plt.figure(figsize=(12, 7))

                # # Plot AUC score
                # plt.plot(
                #     x_days, y_auc, 
                #     color="#1f77b4", linewidth=2.5,
                #     label="D3 AUC Score"
                # )

                # Base line (below threshold)
                plt.plot(x_days, y_auc, color='gray', linewidth=2, alpha=0.5, label='D3 AUC Score')

                # Overlay the bold section above the threshold
                plt.plot(np.array(x_days)[np.array(y_auc) > threshold], np.array(y_auc)[np.array(y_auc) > threshold],
                    color='blue', linewidth=4, label='Above Threshold')

                # Add threshold
                plt.axhline(
                    y=1 - dd_params[0], 
                    color="lightblue", linestyle="--", linewidth=2,
                    label="Drift Detection Threshold"
                )

                # Add vertical lines
                plt.axvline(
                    x=(split_point - 2 * week_len) / day_len, 
                    color="blue", linestyle="--", linewidth=2,
                    label="Leak Start"
                )
                plt.axvline(
                    x=(split_point - week_len) / day_len, 
                    color="#1f77b4", linestyle=":", linewidth=2,
                    label="Max Sample Separation"
                )

                # Highlight drift window
                plt.axvspan(
                    (split_point - 2 * week_len) / day_len,
                    (split_point) / day_len,
                    color="green", alpha=0.3,
                    label="Drift Period"
                )

                # Labels and title
                plt.xlabel("Time (days)", fontsize=14)
                plt.ylabel("1 - AUC Score", fontsize=14)
                plt.title(
                    f"Drift Detection for Leak Node {leak_node} (Diameter: {leak_diam_mm} mm)", 
                    fontsize=16, fontweight="bold", pad=20
                )

                # Legend and grid
                plt.legend(
                    fontsize=12, fancybox=True, loc="upper right"
                )
                plt.grid(True, which="major", linestyle="--", alpha=0.6)

                # Ticks
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)

                # fixed y-axis limits
                plt.ylim(0.4, 1)

                # Optional: tighter layout & export
                plt.tight_layout()
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
with open("top2_shortest_paths_to_sensors.json", "r") as f:
    closest_sensor_info = json.load(f)

# read baseline file
df_baseline = pd.read_csv('Baseline.csv')
# remove timestamp column
df_baseline = df_baseline.drop(columns=['Unnamed: 0'])

n_repeats = 1   # number of repetitions of the experiment for each leaking pipe


# 389 leak nodes
leak_nodes = ['n3','n4','n5','n6','n10','n11','n14','n15','n17','n18','n22','n23',
 'n25','n26','n32','n33','n34','n37','n38','n40','n41','n43','n44','n46',
 'n48','n50','n51','n52','n58','n61','n63','n64','n67','n68','n71','n72',
 'n74','n75','n77','n78','n79','n82','n83','n86','n87','n91','n96','n97',
 'n98','n100','n101','n104','n105','n106','n107','n111','n112','n113','n114','n115',
 'n117','n121','n122','n125','n128','n129','n131','n132','n133','n135','n137','n140',
 'n142','n143','n144','n147','n149','n151','n152','n153','n156','n157','n159','n163',
 'n164','n165','n166','n169','n171','n172','n174','n177','n180','n182','n183','n187',
 'n188','n189','n191','n192','n193','n194','n198','n199','n201','n203','n205','n209',
 'n212','n213','n217','n219','n222','n223','n227','n229','n232','n233','n234','n238',
 'n239','n242','n243','n244','n245','n248','n253','n255','n256','n257','n258','n261',
 'n262','n263','n264','n265','n268','n269','n271','n275','n276','n277','n279','n281',
 'n283','n285','n286','n287','n288','n290','n292','n294','n298','n299','n300','n302',
 'n303','n306','n307','n309','n310','n311','n313','n314','n315','n320','n321','n323',
 'n325','n327','n329','n330','n331','n333','n335','n336','n338','n339','n340','n345',
 'n346','n347','n349','n351','n354','n356','n358','n360','n363','n366','n371','n372',
 'n373','n375','n378','n380','n381','n383','n388','n390','n392','n394','n396','n398',
 'n401','n403','n405','n408','n409','n410','n412','n414','n416','n418','n421','n423',
 'n426','n427','n429','n431','n433','n435','n439','n441','n444','n445','n446','n448',
 'n451','n452','n455','n457','n459','n461','n463','n465','n466','n469','n471','n476',
 'n478','n480','n485','n486','n487','n489','n493','n497','n499','n500','n502','n505',
 'n507','n509','n510','n513','n514','n516','n518','n521','n522','n525','n526','n529',
 'n534','n536','n537','n538','n541','n542','n545','n547','n549','n552','n553','n555',
 'n556','n557','n559','n561','n564','n566','n568','n570','n572','n573','n575','n577',
 'n578','n580','n582','n584','n587','n590','n591','n592','n594','n596','n598','n600',
 'n603','n605','n607','n609','n611','n613','n614','n615','n617','n619','n621','n622',
 'n623','n624','n625','n626','n627','n628','n632','n634','n636','n638','n640','n643',
 'n646','n648','n649','n651','n652','n653','n655','n657','n658','n659','n661','n663',
 'n665','n667','n669','n671','n673','n675','n676','n680','n682','n685','n687','n688',
 'n690','n691','n693','n695','n697','n699','n701','n704','n709','n711','n714','n716',
 'n719','n721','n722','n725','n726','n727','n729','n731','n734','n735','n737','n739',
 'n742','n744','n747','n749','n751','n753','n755','n757','n760','n763','n765','n767',
 'n769','n772','n779','n781','n782']

leak_nodes = ['n44', 'n603']



dd_methods = [d3]   # drift detection methods to evaluate
# dd_methods = [d3, knn_d3_20, knn_d3_50, knn_d3_100]   # drift detection methods to evaluate


# parameters for drift detection methods
# d3 --> 0.4
# ks --> 0.825


# run experiments

# Define the leak sizes you want to test
leak_diameters = [7, 11, 15]

# List to store results for each leak size
all_results = []

for leak_diam_mm in leak_diameters:
    print(f"Running experiments for leak diameter = {leak_diam_mm} mm...")
    
    # Run your existing experiment
    results_df = run_dd_experiments(
        leak_nodes,
        leak_diam_mm,
        n_repeats,
        closest_sensor_info,
        df_baseline,
        dd_methods,
        dd_params=[0.4]  # example thresholds for d3, knn_d3, ks
    )
    
    all_results.append(results_df)

# Concatenate all results into one DataFrame
results_all = pd.concat(all_results, ignore_index=True)

# save results to CSV
save_results = False
if save_results:
    results_all.to_csv("dd_experiment_results.csv", index=False)

# %%
