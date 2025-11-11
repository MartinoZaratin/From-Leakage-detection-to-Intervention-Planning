
import geopandas as gpd
import pandas as pd
import networkx as nx
import json

# Load shapefiles
nodes = gpd.read_file("..\\_shapefiles_\\L-TOWN_Real_junctions.shp")
pipes = gpd.read_file("..\\_shapefiles_\\L-TOWN_Real_pipes.shp")

# read csv for direction instructions
directions = pd.read_csv('flow_directions.csv')
# rename pipe column
directions = directions.rename(columns={'Unnamed: 0': 'pipe_id'})
# remove last four rows: pump and valves
directions = directions[:-4]

# override nodeto and nodefrom in pipes according to directions
pipes[["nodefrom", "nodeto"]] = directions[["From_Node", "To_Node"]].values

# Build graph
G = nx.DiGraph()
for _, row in pipes.iterrows():
    G.add_edge(row["nodefrom"], row["nodeto"], weight=row["length"])

# Define sensors (subset of node IDs)
sensors = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
            'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
            'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
            'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
            'n726', 'n740', 'n752', 'n769']

# Compute distances and paths
lengths, paths = nx.multi_source_dijkstra(G.reverse(), sources=sensors, weight="weight")   # lengths and paths are dicts

# Prepare results
closest_sensors = {}
distances = {}
for node in G.nodes():
    if node in lengths:
        distances[node] = lengths[node]
        closest_sensors[node] = paths[node][0]  # First node in path is the closest sensor
    else:
        distances[node] = float('inf')
        closest_sensors[node] = None


# count NAs in closest_sensors
na_count = sum(1 for sensor in closest_sensors.values() if sensor is None)
na_nodes = [node for node, sensor in closest_sensors.items() if sensor is None]

# load top2_shortest_paths_to_sensors json
with open("top2_shortest_paths_to_sensors.json", "r") as f:
    top2_data = json.load(f)

# check if top2 has nas
top2_na_count = sum(1 for node in na_nodes if top2_data[node]["closest_sensors"] is None)   # = 0

# fill nas with top2 info
for node in na_nodes:
    closest_sensors[node] = top2_data[node]["closest_sensors"][0]
    distances[node] = top2_data[node]["distances"][0]

combined_results = {
    node: {
        "closest_sensor": closest_sensors[node],
        "distance": distances[node]
    }
    for node in G.nodes()
}

# save combined results to json
with open("closest_downstream_sensors.json", "w") as f:
    json.dump(combined_results, f, indent=4)



# add information to node shapefiles
nodes["downstream_sensor"] = nodes["id"].map(closest_sensors)
nodes["distance_to_sensor"] = nodes["id"].map(distances)
# save updated shapefile
nodes.to_file("L-TOWN_Real_junctions_with_flow.shp")
