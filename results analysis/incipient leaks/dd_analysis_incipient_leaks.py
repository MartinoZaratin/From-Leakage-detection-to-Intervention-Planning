#%%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

#%%

# read csv 
results_7mm = pd.read_csv("dd_incipient_leaks_experiment_results_7mm.csv")
results_11mm = pd.read_csv("dd_incipient_leaks_experiment_results_11mm.csv")
results_15mm = pd.read_csv("dd_incipient_leaks_experiment_results_15mm.csv")
results_19mm = pd.read_csv("dd_incipient_leaks_experiment_results_19mm.csv")

results = [results_7mm, results_11mm, results_15mm, results_19mm]


################################################
# PLOT DVELOPMENT TIME VS AUC SCORES
################################################
# %%
# plot development time vs AUC score with mean and std
colors = ['darkorange', 'teal']
leak_sizes = [7, 11, 15, 19]
for idx, results_df in enumerate(results):
    plt.figure(figsize=(10, 6))
    
    for id, dd_method in enumerate(["d3", "ks"]):
        # Group by incipiency duration and calculate mean and std
        grouped = results_df.groupby("incipiency_duration_days")[f"dd_auc_{dd_method}"]
        mean_auc = grouped.mean()
        std_auc = grouped.std()

        # apply rolling average for smoothing
        window = 3  # or 5, depending on how much smoothing you want
        mean_smooth = mean_auc.rolling(window, center=True).mean()
        std_smooth = std_auc.rolling(window, center=True).mean()

        plt.plot(mean_smooth.index, mean_smooth.values, color=colors[id], label=f"{dd_method.upper()}")
        plt.fill_between(mean_smooth.index,
                        mean_smooth.values - std_smooth.values,
                        mean_smooth.values + std_smooth.values,
                        color=colors[id],
                        alpha=0.2)

    plt.xlabel("Development time (days)", fontsize=27)
    plt.ylabel("AUC Score", fontsize=27)
    plt.legend(loc="upper right", fontsize=27)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0.4, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save high-res image
    plt.savefig(f"development_time_vs_auc_{leak_sizes[idx]}.png", dpi=300, bbox_inches='tight')

    plt.show()
# %%
