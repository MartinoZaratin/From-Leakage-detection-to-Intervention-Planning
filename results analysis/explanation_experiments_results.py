#%%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import ast

# %%
# models
def FeatureImportanceRF(X, y, model):
    # Placeholder for actual FeatureImportance implementation
    pass

def FeatureImportanceET(X, y, model):
    # Placeholder for actual FeatureImportance implementation
    pass

def PermutationFeatureImportanceRF(X, y, model):
    # Placeholder for actual PermutationFeatureImportance implementation
    pass

def PermutationFeatureImportanceET(X, y, model):
    # Placeholder for actual PermutationFeatureImportance implementation
    pass

explanation_methods = [FeatureImportanceRF, FeatureImportanceET, 
                       PermutationFeatureImportanceRF, PermutationFeatureImportanceET]


#%%

# read results
results_7mm = pd.read_csv("explanation_experiments_results_7mm.csv")
results_11mm = pd.read_csv("explanation_experiments_results_11mm.csv")
results_15mm = pd.read_csv("explanation_experiments_results_15mm.csv")
results_19mm = pd.read_csv("explanation_experiments_results_19mm.csv")

# combine results
results = pd.concat([results_7mm, results_11mm, results_15mm, results_19mm], ignore_index=True)

# read closest_downstream_sensors json
with open("closest_downstream_sensors.json") as f:
    closest_downstream_sensors = json.load(f)

# map closest downstream sensors to results
results['closest_downstream_sensor'] = results['leak_node'].map(lambda x: closest_downstream_sensors[x]['closest_sensor'])

sensor_nodes = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
                      'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
                      'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
                      'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                      'n726', 'n740', 'n752', 'n769']

#############################
# PRINT ACCURACY SCORES
#############################
# %%
# closest sensor
# print for each methond and leak size
for method in explanation_methods:
    for leak_size in [7, 11, 15, 19]:
        subset = results[results['leak_diam_mm'] == leak_size]
        accuracy = sum(subset[f'best_explained_sensor_{method.__name__}'] == subset['closest_sensor']) / subset.shape[0]
        print(f"Method: {method.__name__}, Leak Size: {leak_size}mm, Accuracy: {accuracy:.4f}")
    print("\n")


# %%
#############################
# PLOT ACCURACY SCORES
#############################
# %%
# plot accuracy for closest sensor
plt.figure(figsize=(10, 6))
for method in explanation_methods:
    # Group results by leak size
    grouped = results.groupby("leak_diam_mm")[f"best_explained_sensor_{method.__name__}"]
    accuracy = grouped.apply(lambda x: np.mean(x == results.loc[x.index, 'closest_sensor']))

    # Plot accuracy
    plt.plot(accuracy.index, accuracy.values, marker='o', label=method.__name__)

plt.xlabel("Leak Diameter (mm)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
# plt.title("Explanation Method Accuracy vs Leak Diameter", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.ylim(0, 1)
plt.grid(True)
# Save high-res image
# plt.savefig("explanation_accuracy_plot.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
# plot acccuracy for closest and closest downstream sensor
plt.figure(figsize=(10, 10))
for method in explanation_methods:
    # Group results by leak size
    grouped = results.groupby("leak_diam_mm")[f"best_explained_sensor_{method.__name__}"]
    accuracy = grouped.apply(lambda x: np.mean((x == results.loc[x.index, 'closest_sensor']) |
                                               (x == results.loc[x.index, 'closest_downstream_sensor'])))

    # Plot accuracy
    plt.plot(accuracy.index, accuracy.values, label=method.__name__)
plt.xlabel("Leak Diameter (mm)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
# plt.title("Explanation Method Accuracy vs Leak Diameter", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.ylim(0, 1)
plt.grid(True)

# %%
##############################
# PLOT EXPLANATION SCORES
##############################
# choose a random node and plot explanation scores for different methods
leak_node = 'n658'
leak_diam_mm = 15
colors = ['gray'] * 32 + ['navy']

for method in [PermutationFeatureImportanceRF, PermutationFeatureImportanceET]:
    explanation_scores = ast.literal_eval(results.loc[results['leak_node'] == leak_node, f'feature_scores_{method.__name__}'].values[0])
    # transform scores to float
    explanation_scores = {sensor: float(score) for sensor, score in explanation_scores.items()}
    # sort dictionary by scores ascending
    explanation_scores = dict(sorted(explanation_scores.items(), key=lambda item: item[1]))
    # horizontal histogram of explanation scores
    plt.figure(figsize=(6, 10))
    plt.barh(list(explanation_scores.keys()), list(explanation_scores.values()), color=colors, zorder=2)
    plt.xlabel("Explanation score", fontsize=20)
    plt.ylabel("Sensor node", fontsize=20)
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save high-res image
    plt.savefig(f"explanation_scores_{method.__name__}_leak_{leak_node}_diam_{leak_diam_mm}mm.png", dpi=300, bbox_inches='tight')
    plt.show()


# for method in explanation_methods:
#     explanation_scores = ast.literal_eval(results.loc[results['leak_node'] == leak_node, f'feature_scores_{method.__name__}'].values[0])
#     scores = [explanation_scores[sensor] for sensor in sensor_nodes]
    
#     plt.plot(sensor_nodes, scores, marker='o', label=method.__name__)
# %%

#####################################
# CONFUSION MATRICES FOR BEST EXPLAINED SENSORS
######################################
import seaborn as sns
from sklearn.metrics import confusion_matrix
# plot for the whole dataset, leaksize = 15mm
leak_diam_mm = 15
results_subset = results[results['leak_diam_mm'] == leak_diam_mm] 
for method in [PermutationFeatureImportanceET]:
    y_true = results_subset['closest_sensor']
    y_pred = results_subset[f'best_explained_sensor_{method.__name__}']
    
    cm = confusion_matrix(y_true, y_pred, labels=sensor_nodes)
    
    # annot = np.where(cm != 0, cm.astype(str), "")

    plt.figure(figsize=(14,12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako_r', xticklabels=sensor_nodes, yticklabels=sensor_nodes)
    plt.xlabel('Predicted sensor', fontsize=30)
    plt.ylabel('True closest sensor', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Save high-res image
    plt.savefig(f'confusion_matrix_{method.__name__}_leak_diam_{leak_diam_mm}mm.png', dpi=200, bbox_inches='tight')
    plt.show()

# %%
