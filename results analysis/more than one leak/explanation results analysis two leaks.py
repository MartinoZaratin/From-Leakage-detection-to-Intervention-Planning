#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import ast

#%%

df =  pd.read_csv('explanation_results_two_leaks.csv')

# read json file with closest sensors
with open('top2_shortest_paths_to_sensors.json', 'r') as f:
    closest_sensors = json.load(f)
# %%
for index, row in df.loc[:10, :].iterrows():
    leak1 = row['leak_node_1']
    leak2 = row['leak_node_2']
    closest_sensor1 = closest_sensors[leak1]['closest_sensors'][0]
    closest_sensor2 = closest_sensors[leak2]['closest_sensors'][0]

    feature_scores = ast.literal_eval(row['feature_scores_PermutationFeatureImportanceET'])
    # sort feature scores in descending order
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    # find ranking of closest sensors
    ranking_sensor1 = next((i for i, v in enumerate(sorted_features) if v[0] == closest_sensor1), None)
    ranking_sensor2 = next((i for i, v in enumerate(sorted_features) if v[0] == closest_sensor2), None)

    print(f"Leak1: {leak1}, Closest Sensor1: {closest_sensor1}, Ranking: {ranking_sensor1+1}")
    print(f"Leak2: {leak2}, Closest Sensor2: {closest_sensor2}, Ranking: {ranking_sensor2+1}")

# %%
# retireve scores for all rows
rankings_sensor1 = []
rankings_sensor2 = []
for index, row in df.iterrows():
    leak1 = row['leak_node_1']
    leak2 = row['leak_node_2']
    closest_sensor1 = closest_sensors[leak1]['closest_sensors'][0]
    closest_sensor2 = closest_sensors[leak2]['closest_sensors'][0]

    feature_scores = ast.literal_eval(row['feature_scores_PermutationFeatureImportanceET'])
    # sort feature scores in descending order
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    # find ranking of closest sensors
    ranking_sensor1 = next((i for i, v in enumerate(sorted_features) if v[0] == closest_sensor1), None)
    ranking_sensor2 = next((i for i, v in enumerate(sorted_features) if v[0] == closest_sensor2), None)

    rankings_sensor1.append(ranking_sensor1 + 1)  # +1 for 1-based ranking
    rankings_sensor2.append(ranking_sensor2 + 1)  # +1 for 1-based ranking

# %%
# compute metrics
# EXACT ACCURACY
# perfect score is 1,1 - 1,2 - 2,1 - 2,2
exact_matches = sum(1 for r1, r2 in zip(rankings_sensor1, rankings_sensor2) if (r1 <= 2 and r2 <= 2))
exact_accuracy = exact_matches / len(df)
print(f"Exact Accuracy: {exact_accuracy:.4f}")

# TOP 3 ACCURACY
top3_matches = sum(1 for r1, r2 in zip(rankings_sensor1, rankings_sensor2) if (r1 <= 3 and r2 <= 3))
top3_accuracy = top3_matches / len(df)
print(f"Top 3 Accuracy: {top3_accuracy:.4f}")

# TOP 5 ACCURACY
top5_matches = sum(1 for r1, r2 in zip(rankings_sensor1, rankings_sensor2) if (r1 <= 5 and r2 <= 5))
top5_accuracy = top5_matches / len(df)
print(f"Top 5 Accuracy: {top5_accuracy:.4f}")

# MARGINAL TOP 1 ACCURACY
marginal_top1_matches = sum(1 for r1, r2 in zip(rankings_sensor1, rankings_sensor2) if (r1 == 1 or r2 == 1))
marginal_top1_accuracy = marginal_top1_matches / len(df)
print(f"Marginal Top 1 Accuracy: {marginal_top1_accuracy:.4f}")

# MEAN RECIPROCAL RANK
mrr = np.mean([ (1/r1 + 1/r2)/2 for r1, r2 in zip(rankings_sensor1, rankings_sensor2)])
print(f"Mean Reciprocal Rank: {mrr:.4f}")

# MEAN RANKS for each sensor
mean_rank_sensor1 = np.mean(rankings_sensor1)
mean_rank_sensor2 = np.mean(rankings_sensor2)
print(f"Mean Rank Sensor 1: {mean_rank_sensor1:.2f}")
print(f"Mean Rank Sensor 2: {mean_rank_sensor2:.2f}")


# %%
