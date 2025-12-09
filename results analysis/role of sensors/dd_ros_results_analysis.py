#%%
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

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
# results = pd.read_csv("dd_ros_25_sensors_results.csv")
results_9_sensors = pd.read_csv("dd_ros_9_sensors_results.csv")
results_17_sensors = pd.read_csv("dd_ros_17_sensors_results.csv")
results_25_sensors = pd.read_csv("dd_ros_25_sensors_results.csv")
results_33_sensors = pd.read_csv("dd_experiment_results.csv")
results_33_sensors_19mm = pd.read_csv("dd_experiment_results_19mm.csv")
results_33_sensors = pd.concat([results_33_sensors, results_33_sensors_19mm], ignore_index=True)

dd_methods = [d3]

dd_names = ['D3']
colors = ['darkorange']


# plot number of sensors vs mean auc score. One line per leak size. Shaded stddev
leak_sizes = [7, 11, 15, 19]
for leak_size in leak_sizes:
    mean_aucs = []
    std_aucs = []
    for results in [results_9_sensors, results_17_sensors, results_25_sensors, results_33_sensors]:
        mean_auc = results[results["leak_diam_mm"] == leak_size][f"dd_auc_d3"].mean()
        std_auc = results[results["leak_diam_mm"] == leak_size][f"dd_auc_d3"].std()
        mean_aucs.append(mean_auc)
        std_aucs.append(std_auc)
    
    plt.figure(figsize=(10, 6))
    plt.plot([9, 17, 25, 33], mean_aucs, marker='o', label='D3', color='darkorange')
    plt.fill_between([9, 17, 25, 33],
                     np.array(mean_aucs) - np.array(std_aucs),
                     np.array(mean_aucs) + np.array(std_aucs),
                     color = 'darkorange', alpha=0.2)
    plt.xlabel("Number of Sensors", fontsize=27)
    plt.ylabel("AUC Score", fontsize=27)
    plt.legend(loc="lower right", fontsize=27)
    plt.xticks([9, 17, 25, 33], fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0.4, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"dd_auc_vs_sensors_leak_{leak_size}mm.png", dpi=200, bbox_inches='tight')
    plt.show()

    
#%%
results_without_closest = pd.read_csv("dd_ros_without_closest_sensors_results.csv")
# plot leak size vs mean auc score for results without closest sensors
leak_sizes = [7, 11, 15, 19]
mean_aucs = []
std_aucs = []
for leak_size in leak_sizes:
    mean_auc = results_without_closest[results_without_closest["leak_diam_mm"] == leak_size][f"dd_auc_d3"].mean()
    std_auc = results_without_closest[results_without_closest["leak_diam_mm"] == leak_size][f"dd_auc_d3"].std()
    mean_aucs.append(mean_auc)
    std_aucs.append(std_auc)
plt.figure(figsize=(10, 6))
plt.plot(leak_sizes, mean_aucs, marker='o', label='Mean D3 AUC Score', color='darkorange')
plt.fill_between(leak_sizes,
                 np.array(mean_aucs) - np.array(std_aucs),
                 np.array(mean_aucs) + np.array(std_aucs),
                 color = 'darkorange', alpha=0.2)
plt.xlabel("Leak Size (mm)", fontsize=12)
plt.ylabel("AUC Score", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.ylim(0.4, 1)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig(f"dd_auc_vs_leak_size_without_closest_sensors.png", dpi=200, bbox_inches='tight')
plt.show()


#%%
###############################################
# CREATE HEATMAP FOR DRIFT DETECTION AUC SCORES
###############################################
# load results
results = pd.read_csv("dd_ros_17_sensors_results.csv")

# load pipes, nodes shape files
pipes = gpd.read_file("..\\..\\_shapefiles_\\L-TOWN_Real_pipes.shp")
nodes = gpd.read_file("..\\..\\_shapefiles_\\L-TOWN_Real_junctions.shp")

# modify pump connection
pipes.loc[pipes['id'] == 'p239', 'nodefrom'] = 'n54'
pipes.loc[pipes['id'] == 'p239', 'length'] = 90.0


# add reservoirs to nodes as new nodes
reservoirs = [{'id': 'R1'}, {'id': 'R2'}]
nodes = pd.concat([nodes, pd.DataFrame(reservoirs)], ignore_index=True)


# map dd_auc_d3 column to nodes according to leak_node column. If no leak at node, set to NaN
for leak_diam_mm in [7, 11, 15, 19]:
    nodes[f"dd_auc_d3_leak_{leak_diam_mm}"] = np.nan
    pipes[f"dd_auc_d3_leak_{leak_diam_mm}"] = np.nan   # initialize column in pipes as well
    for _, row in results[results["leak_diam_mm"] == leak_diam_mm].iterrows():
        leak_node = row["leak_node"]
        auc_score = row["dd_auc_d3"]
        nodes.loc[nodes["id"] == leak_node, f"dd_auc_d3_leak_{leak_diam_mm}"] = auc_score


import pandas as pd
import numpy as np
from collections import deque

def propagate_nodes_to_pipes(pipes: pd.DataFrame, nodes: pd.DataFrame,
                             leak_field_template="dd_auc_d3_leak_{}",
                             leak_diams=(7,11,15,19)):
    """
    Mutates pipes and nodes DataFrames in-place: fills NaNs in the leak fields
    propagating values from selected nodes through the network.
    """
    # ensure id columns exist and are unique
    assert 'id' in pipes.columns and 'id' in nodes.columns
    assert 'nodefrom' in pipes.columns and 'nodeto' in pipes.columns

    # build adjacency: node_id -> list of (neighbor_node, pipe_id)
    adj = {}
    for _, row in pipes[['id','nodefrom','nodeto']].iterrows():
        pid, a, b = row['id'], row['nodefrom'], row['nodeto']
        adj.setdefault(a, []).append((b, pid))
        adj.setdefault(b, []).append((a, pid))

    # helper to write back dicts to DataFrames at the end
    pipes_index = {pid: i for i, pid in enumerate(pipes['id'].values)}
    nodes_index = {nid: i for i, nid in enumerate(nodes['id'].values)}

    for leak in leak_diams:
        field = leak_field_template.format(leak)
        # ensure the columns exist
        if field not in nodes.columns:
            nodes[field] = np.nan
        if field not in pipes.columns:
            pipes[field] = np.nan

        # build fast lookup dicts from current dataframe values
        node_vals = {}
        for _, r in nodes[['id', field]].iterrows():
            val = r[field]
            if not pd.isna(val):
                node_vals[r['id']] = float(val)

        pipe_vals = {}
        for _, r in pipes[['id', field]].iterrows():
            val = r[field]
            if not pd.isna(val):
                pipe_vals[r['id']] = float(val)

        # queue seeded with all nodes that already have a value
        q = deque(node_vals.keys())

        # BFS-like propagation
        while q:
            u = q.popleft()
            u_val = node_vals.get(u)
            # iterate incident edges
            for v, pid in adj.get(u, []):
                # current known values
                p_val = pipe_vals.get(pid)
                v_val = node_vals.get(v)

                # if pipe already has value, maybe it can help node v:
                if p_val is not None:
                    if v_val is None:
                        # give the node the pipe's value
                        node_vals[v] = p_val
                        q.append(v)
                    continue  # pipe already set, next edge

                # pipe not set yet -> decide value
                if v_val is not None:
                    # both nodes have values -> mean
                    mean_val = (u_val + v_val) / 2.0
                    pipe_vals[pid] = mean_val
                    # propagate pipe -> any node missing
                    if pd.isna(nodes.loc[nodes_index[v], field]):
                        nodes.iat[nodes_index[v], nodes.columns.get_loc(field)] = mean_val
                    if pd.isna(nodes.loc[nodes_index[u], field]):
                        nodes.iat[nodes_index[u], nodes.columns.get_loc(field)] = mean_val
                    # if v had a value it's already in queue or processed; but ensure u/v propagate
                    # add both nodes to queue in case neighbors of v or u benefit
                    q.append(v)
                    q.append(u)
                else:
                    # only u has value -> pipe takes u_val and propagate to v
                    pipe_vals[pid] = u_val
                    if v not in node_vals:
                        node_vals[v] = u_val
                        q.append(v)
                    # also ensure pipe to df later

        # done propagation for this leak diameter
        # write pipe_vals and node_vals back to dataframes (only for entries we filled)
        # update pipes
        if pipe_vals:
            # create a series mapping index -> value and assign directly for speed
            pipe_idx_vals = {pipes_index[pid]: val for pid, val in pipe_vals.items()}
            # assign
            for idx, val in pipe_idx_vals.items():
                pipes.iat[idx, pipes.columns.get_loc(field)] = val

        # update nodes
        if node_vals:
            node_idx_vals = {nodes_index[nid]: val for nid, val in node_vals.items()}
            for idx, val in node_idx_vals.items():
                nodes.iat[idx, nodes.columns.get_loc(field)] = val

    return pipes, nodes

processed_pipes, processed_nodes = propagate_nodes_to_pipes(pipes, nodes)


# save processed pipes to new shapefile
processed_pipes.to_file("processed_pipes_17_sensors.shp")

# %%
