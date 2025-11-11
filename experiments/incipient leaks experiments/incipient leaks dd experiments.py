# --------------------
# Load libraries
# --------------------
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
# import seaborn as sns

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
#   EXPERIMENT FUNCTIONS
#-----------------------
def generate_incipient_leak(baseline_df, leak_df, step_start, step_full):
    """
    Generate an incipient leak scenario by linearly blending between baseline and full-leak data.

    baseline_df, leak_df: DataFrames with identical index (timesteps) and same columns (nodes).
    step_start: timestep index when leak begins.
    step_full: timestep index when leak reaches full size.
    """
    df = baseline_df.copy()
    # trim to the length of leak_df
    df = df[:len(leak_df)]
    n_steps = len(leak_df)
    weights = np.zeros(n_steps, dtype=float)

    for i in range(n_steps):
        if i < step_start:
            weights[i] = 0
        elif step_start <= i <= step_full:
            weights[i] = (i - step_start) / (step_full - step_start)
        else:
            weights[i] = 1

    # Apply blending across all nodes
    df[:] = df * (1 - weights[:, None]) + leak_df * weights[:, None]
    return df


def run_dd_experiments(leak_nodes, leak_diam_mm, n_repeats, closest_sensor_info, df_baseline, dd_methods, dd_params):
    results = []

    for leak_node in leak_nodes:
        # read file with leakage data
        df = pd.read_csv(f'..\\..\\dataset generator\\Datasets\\Leakages {leak_diam_mm}mm\\leakage_{leak_node}_{leak_diam_mm}.csv')
        # remove timestamp column
        df = df.drop(columns=['Unnamed: 0'])

        # generate n_repeats random duration of incipiency for the leak between 7 and 45 days
        incipiency_durations = [random.randint(7, 45) for _ in range(n_repeats)]
        
        n_timesteps = df.shape[0]
        for id, incipiency_duration in enumerate(incipiency_durations):
            # generate random t_start in the possible range
            max_start = n_timesteps - incipiency_duration * day_len - week_len
            t_start = random.randint(week_len, max_start)
            t_full = t_start + incipiency_duration * day_len

            # generate incipient leak data
            data = generate_incipient_leak(df_baseline.values, df.values, t_start, t_full)

            
            print(f"Running experiment for leak node {leak_node}, repetition {id+1}/{n_repeats}...")

            # initialize record
            record = {"leak_node": leak_node,
                    "leak_diam_mm": leak_diam_mm,
                    "leak_start_at": t_start * 15 * 60,     # in seconds
                    "full_leak_at": t_full * 15 * 60, # in seconds
                    "incipiency_duration_days": incipiency_duration,
                    "closest_sensor": closest_sensor_info[leak_node]["closest_sensors"][0]}

            # 1. DRIFT DETECTION
            # run drift detection on sliding windows (one check every day, windows of two weeks)
            for j, dd_method in enumerate(dd_methods):
                i = 0
                dd_results = []
                while i + 2*week_len < data.shape[0]:
                    X = data[i:i+2*week_len, :]
                    dd_result = dd_method(X)
                    dd_results.append((dd_result, (i + 2* week_len - incipiency_duration // 2 * day_len >= t_start) and (i + week_len <= t_full)))  # store also if drift is in the window
                                                                                                                              # half duration margin before, 1 week margin after
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
                # sns.set_theme(style="whitegrid", context="talk")

                x_days = np.arange(7, len(dd_results)+7)
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
                plt.axvline(x=(t_start) / day_len, color="green", linestyle="--", linewidth=2, label="Leak start")
                plt.axvline(x=(t_full) / day_len, color="green", linestyle="--", linewidth=2, label="Leak full size")

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
                if save_figs:
                    plt.savefig(f"drift_detection_leak_{leak_node}_diam_{leak_diam_mm}.png", dpi=300)
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
with open("..\\top2_shortest_paths_to_sensors.json", "r") as f:
    closest_sensor_info = json.load(f)

# read baseline file
df_baseline = pd.read_csv('..\\Baseline.csv')
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


dd_methods = [d3, ks]   # drift detection methods to evaluate

# parameters for drift detection methods
# d3 --> 0.4
# ks --> 0.825

# run experiments

# Define the leak sizes
leak_diameters = [7, 11, 15, 19]

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
        dd_params=[0.4, 0.825]  # example thresholds for d3, knn_d3, ks
    )
    
    # save results for this leak size
    results_df.to_csv(f"dd_incipient_leaks_experiment_results_{leak_diam_mm}mm.csv", index=False)

    all_results.append(results_df)

# Concatenate all results into one DataFrame
results_all = pd.concat(all_results, ignore_index=True)

# save results to CSV
save_results = False
if save_results:
    results_all.to_csv("dd_incipient_leaks_experiment_results.csv", index=False)

# %%
