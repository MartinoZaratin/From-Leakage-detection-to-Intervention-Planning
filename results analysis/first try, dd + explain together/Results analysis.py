
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read csv files
results_df_7 = pd.read_csv('experiment_results_leak_7mm.csv')
results_df_11 = pd.read_csv('experiment_results_leak_11mm.csv') 
results_df_15 = pd.read_csv('experiment_results_leak_15mm.csv')

# drift detection plots
# plot abs_value(leak_at - dd_timepoint) for each leak size
# put experiments in the x axis
# plot all dd_timepoint_d3, dd_timepoint_knn_d3, dd_timepoint_ks
leak_sizes = [7, 11, 15]

for leak_size, results_df in zip(leak_sizes, 
                                  [results_df_7, results_df_11, results_df_15]):
    abs_diffs_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_d3']) / 3600  # convert to hours
    abs_diffs_knn_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_knn_d3']) / 3600
    abs_diffs_ks = np.abs(results_df['leak_at'] - results_df['dd_timepoint_ks']) / 3600
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df.index, abs_diffs_d3, marker='o', label='DD Timepoint D3')
    plt.plot(results_df.index, abs_diffs_knn_d3, marker='s', label='DD Timepoint KNN D3')
    plt.plot(results_df.index, abs_diffs_ks, marker='^', label='DD Timepoint KS')
    
    plt.title(f'Absolute Difference between Leak At and DD Timepoint (Leak Size: {leak_size}mm)')
    plt.xlabel('Experiment Index')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid()
    plt.show()


from sklearn.metrics import accuracy_score
for leak_size, results_df in zip(leak_sizes, 
                                  [results_df_7, results_df_11, results_df_15]):
    y_true = [1]*len(results_df)  # all have leaks
    y_pred_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_d3']) / 3600 < 24
    y_pred_knn_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_knn_d3']) / 3600 < 24
    y_pred_ks = np.abs(results_df['leak_at'] - results_df['dd_timepoint_ks']) / 3600 < 24
    
    acc_d3 = accuracy_score(y_true, y_pred_d3)
    acc_knn_d3 = accuracy_score(y_true, y_pred_knn_d3)
    acc_ks = accuracy_score(y_true, y_pred_ks)
    
    print(f'Leak Size: {leak_size}mm')
    print(f'Accuracy DD Timepoint D3: {acc_d3:.2f}')
    print(f'Accuracy DD Timepoint KNN D3: {acc_knn_d3:.2f}')
    print(f'Accuracy DD Timepoint KS: {acc_ks:.2f}')
    print('-----------------------------------')

# plot accuracies: leak sizes in x axis, accuracies in y axis, one line for each method
accuracies_d3 = []
accuracies_knn_d3 = []
accuracies_ks = []
for leak_size, results_df in zip(leak_sizes, 
                                  [results_df_7, results_df_11, results_df_15]):
    y_true = [1]*len(results_df)  # all have leaks
    y_pred_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_d3']) / 3600 < 24
    y_pred_knn_d3 = np.abs(results_df['leak_at'] - results_df['dd_timepoint_knn_d3']) / 3600 < 24
    y_pred_ks = np.abs(results_df['leak_at'] - results_df['dd_timepoint_ks']) / 3600 < 24
    
    acc_d3 = accuracy_score(y_true, y_pred_d3)
    acc_knn_d3 = accuracy_score(y_true, y_pred_knn_d3)
    acc_ks = accuracy_score(y_true, y_pred_ks)
    
    accuracies_d3.append(acc_d3)
    accuracies_knn_d3.append(acc_knn_d3)
    accuracies_ks.append(acc_ks)

plt.figure(figsize=(10, 6))
plt.plot(leak_sizes, accuracies_d3, marker='o', label='DD Timepoint D3')
plt.plot(leak_sizes, accuracies_knn_d3, marker='s', label='DD Timepoint KNN D3')
plt.plot(leak_sizes, accuracies_ks, marker='^', label='DD Timepoint KS')
plt.title('Accuracy of Drift Detection Methods by Leak Size')
plt.xlabel('Leak Size (mm)')
plt.ylabel('Accuracy')
plt.xticks(leak_sizes)
plt.legend()
plt.grid()
plt.show()

