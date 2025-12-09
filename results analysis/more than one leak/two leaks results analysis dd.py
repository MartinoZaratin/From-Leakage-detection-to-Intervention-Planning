#%%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_mean_std(x, y, bins):   # once a day
    x = np.array(x)
    y = np.array(y)

    # Create bins
    if isinstance(bins, int):
        bins = np.linspace(x.min(), x.max(), bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    means = []
    stds = []

    # Compute statistics per bin
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        y_bin = y[mask]
        if len(y_bin) > 0:
            means.append(np.mean(y_bin))
            stds.append(np.std(y_bin))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    means = np.array(means)
    stds = np.array(stds)
    return bin_centers, means, stds


#%%
####################################
# PLOT DRIFT DETECTION AUC SCORES
####################################

def d3(X, clf=None):
    # Placeholder for the actual d3 implementation
    pass

def ks(X):
    # Placeholder for the actual ks implementation
    pass

# load csv results
results = pd.read_csv("dd_results_two_leaks.csv")

dd_methods = [d3, ks]

dd_names = ['D3', 'KS']
colors = ['darkorange', 'teal']
plt.figure(figsize=(10, 6))

for id, dd_method in enumerate(dd_methods):
    # plot time_displacement vs AUC score
    x = results["time_displacement"]
    y = results[f"dd_auc_{dd_method.__name__}"]
    x_bins, y_mean, y_std = plot_mean_std(x, y, bins=30)
    
    # aplly rolling mean to smooth the curve
    y_mean = pd.Series(y_mean).rolling(window=3, min_periods=1).mean()
    y_std = pd.Series(y_std).rolling(window=3, min_periods=1).mean()

    plt.plot(x_bins / 3600, y_mean, label=dd_names[id], color=colors[id])
    plt.fill_between(x_bins / 3600, y_mean - y_std, y_mean + y_std, color=colors[id], alpha=0.2)


plt.xlabel("Time displacement (hours)", fontsize=12)
plt.ylabel("AUC Score", fontsize=12)
# plt.title("Drift Detection AUC Scores vs Leak Diameter", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.ylim(0.4, 1)
plt.grid(True, linestyle='--', alpha=0.7)

# Save high-res image
plt.savefig("plot.png", dpi=200, bbox_inches='tight')

plt.show()

# %%
