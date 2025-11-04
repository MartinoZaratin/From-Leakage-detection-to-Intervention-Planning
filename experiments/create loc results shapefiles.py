import pandas as pd
import geopandas as gpd
import json

# read selected nodes shapefile
selected_nodes = gpd.read_file("..\\_shapefiles_\\Selected leaks_1.shp")

# remove all column except 'id' and geometry
selected_nodes = selected_nodes[['id', 'geometry']]

# read drift localization results
results = pd.read_csv('experiment_results_leak_15mm.csv')

# keep only leak_node and best explained sensor columns
results = results[['leak_node', 'best_explained_sensor']]

# merge to the gpd
merged = selected_nodes.merge(results, left_on='id', right_on='leak_node')

# remove leak_node column
merged = merged.drop(columns=['leak_node'])

# read json file with sensor proximity information
with open("top2_shortest_paths_to_sensors.json", "r") as f:
    closest_sensor_info = json.load(f)

# map closest sensors to the gpd
merged['closest_sensor'] = merged['id'].map(lambda x: closest_sensor_info[x]['closest_sensors'][0])

# save to shapefile
merged.to_file("..\\_shapefiles_\\Drift localization results.shp")