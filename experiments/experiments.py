#%%
'''
This code runs drift detection and drift localization experiments for a given leakage size
'''
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

#%%
# -------------------
# define methods
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

# ERROR: THE CLOSEST SENSOR HAS BEEN COMPUTED ON THE DISCONNECTED GRAPH
# read json file with sensor proximity information
with open("shortest_paths_to_sensors.json", "r") as f:
    closest_sensor_info = json.load(f)

# read baseline file
df_baseline = pd.read_csv('Baseline.csv')
# remove timestamp column
df_baseline = df_baseline.drop(columns=['Unnamed: 0'])

leak_diam_mm = 15  # leak diameter in mm
n_repeats = 1   # number of repetitions of the experiment for each leaking pipe

drift_localization = True

# define explanation method and parameters
explanation_method = FeatureImportanceRF
explanation_params = {}


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

# leak_nodes = leak_nodes[:10]  # for testing, use only first 10 leak nodes


#-------------------------------------------
# run experiments for a fixed leakage size
# ------------------------------------------
# initialize results list
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

        # time
        start_time = time.time()

        # initialize record
        record = {"leak_node": leak_node,
                  "leak_diam_mm": leak_diam_mm,
                  "leak_at": split_point * 15 * 60,     # in seconds
                  "closest_sensor": closest_sensor_info[leak_node]["closest_sensor"]}

        # concatenate baseline and leak data at split point as numpy array
        data = pd.concat([df_baseline.iloc[:split_point], df.iloc[split_point:]], axis=0).reset_index(drop=True).to_numpy()

        # 1. DRIFT DETECTION
        # run drift detection on sliding windows (one check every day, windows of two weeks)
        i = 0
        dd_results = []
        while i + 2*week_len < data.shape[0]:
            X = data[i:i+2*week_len, :]
            dd_result = d3(X)
            dd_results.append(dd_result)
            i += day_len

        # store results
        record["dd_score"] = min(dd_results)
        record["dd_timepoint"] = (np.argmin(dd_results) + 7 ) * day_len * 15 * 60  # in seconds
                                                                                   # add 7 days to center the window on the detection point


        # optional: visualize results above acertian threshold
        # if min(dd_results) > 0.35:
        #     # plot dd results
        #     plt.figure(figsize=(10,6))
        #     plt.plot(np.arange(len(dd_results))*day_len/ hour_len, dd_results, marker='o')
        #     plt.axvline(x=(split_point - week_len) / hour_len, color='r', linestyle='--', label='Leak Start')
        #     plt.xlabel('Time (hours)')
        #     plt.ylabel('D3 Drift Score')
        #     plt.title(f'Drift Detection Results for Leak Node {leak_node}, Leak Diameter {leak_diam_mm} mm')
        #     plt.legend()
        #     plt.show()


        if drift_localization:
            # 2. DRIFT LOCALIZATION / EXPLANATION
            # retrieve drift time point
            dd_timepoint = (np.argmin(dd_results) + 7 ) * day_len

            # train - test split for explanation
            X = data[dd_timepoint - week_len : dd_timepoint + week_len, :]
            y = np.array(X.shape[0]//2*[0] + X.shape[0]//2*[1])  # first half is class 0, second half is class 1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            # run explanations
            model, feature_scores = explanation_method(X_train, y_train, **explanation_params)
    
            # record time
            end_time = time.time()
            record["execution_time_sec"] = end_time - start_time
        
            # ensure correct shape
            if len(feature_scores.shape) != 1:
                print("Deformed feature scores: '%s' %s"%(explanation_method,str(feature_scores.shape)))
                feature_scores = feature_scores.ravel()
            
            record["explanation_feature_scores"] = {col: f"{score:.5f}" for col, score in zip(df.columns, feature_scores)}
            record['best_explained_sensor'] = df.columns[np.argmax(feature_scores)]
            record['second_best_explained_sensor'] = df.columns[np.argsort(feature_scores)[-2]]

        # append record to results
        results.append(record)


results_df = pd.DataFrame(results)

# convert results to DataFrame and save to CSV
save_results = False
if save_results:
    results_df.to_csv(f'experiment_results_leak_{leak_diam_mm}mm.csv', index=False)



plots = False
if plots:
    # compute distances in time
    distances = np.abs((results_df['leak_at'] - results_df['dd_timepoint']))  # in seconds

    #plot distances against index
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(distances / 3600, marker='o', linestyle='', alpha=0.5)
    plt.xlabel('Experiment Index')
    plt.ylabel('Detection Delay (hours)')
    plt.title(f'Drift Detection Delay for Leak Diameter {leak_diam_mm} mm')
    plt.ylim(0, 100)   # limit y-axis to 100 hours
    plt.grid()
    plt.show()
