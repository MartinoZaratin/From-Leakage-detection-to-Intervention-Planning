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

# combine results
results = pd.concat([results_7mm, results_11mm, results_15mm], ignore_index=True)

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
# PLOT ACCURACY SCORES
#############################
# %%
# closest sensor



# both closest and closest downstream
for method in explanation_methods:
    print(sum((results[f'best_explained_sensor_{method.__name__}'] == results['closest_sensor']) |
                (results[f'best_explained_sensor_{method.__name__}'] == results['closest_downstream_sensor'])
              ) / results.shape[0])

# %%
# plot acccuracy for closest sensor
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
    plt.xlabel("Explanation score", fontsize=12)
    plt.ylabel("Sensor node", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    # Save high-res image
    plt.savefig(f"explanation_scores_{method.__name__}_leak_{leak_node}_diam_{leak_diam_mm}mm.png", dpi=300, bbox_inches='tight')
    plt.show()


# for method in explanation_methods:
#     explanation_scores = ast.literal_eval(results.loc[results['leak_node'] == leak_node, f'feature_scores_{method.__name__}'].values[0])
#     scores = [explanation_scores[sensor] for sensor in sensor_nodes]
    
#     plt.plot(sensor_nodes, scores, marker='o', label=method.__name__)
# %%
