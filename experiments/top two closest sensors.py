
import geopandas as gpd
import networkx as nx
import json

# Load shapefiles
nodes = gpd.read_file("..\_shapefiles_\L-TOWN_Real_junctions.shp")
pipes = gpd.read_file("..\_shapefiles_\L-TOWN_Real_pipes.shp")

# modify pump connection
pipes.loc[pipes['id'] == 'p239', 'nodefrom'] = 'n54'
pipes.loc[pipes['id'] == 'p239', 'length'] = 90.0

# Build graph
G = nx.Graph()
for _, row in pipes.iterrows():
    G.add_edge(row["nodefrom"], row["nodeto"], weight=row["length"])

# Add missing connections corresponding to valves
G.add_edge('n336', 'n111', weight=30.0)
G.add_edge('n226', 'n229', weight=40.0)
G.add_edge('n300', 'n303', weight=25.0)

# Define sensors (subset of node IDs)
sensors = [
    'n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
    'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
    'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
    'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
    'n726', 'n740', 'n752', 'n769'
]

# Compute distances from each sensor
distances_from_sensors = {}
for sensor in sensors:
    lengths = nx.single_source_dijkstra_path_length(G, source=sensor, weight="weight")
    for node, dist in lengths.items():
        if node not in distances_from_sensors:
            distances_from_sensors[node] = []
        distances_from_sensors[node].append((sensor, dist))

# Find top 2 closest sensors for each node
top2_results = {}
for node, sensor_dists in distances_from_sensors.items():
    # Sort by distance
    sorted_sensors = sorted(sensor_dists, key=lambda x: x[1])
    # Take up to 2 closest
    top2_results[node] = {
        "closest_sensors": [s for s, _ in sorted_sensors[:2]],
        "distances": [d for _, d in sorted_sensors[:2]]
    }

# check if any node has less than 2 sensors
for node, info in top2_results.items():
    if len(info["closest_sensors"]) < 2:
        print(f"Node {node} has less than 2 sensors: {info}")

# Save results to JSON
with open("top2_shortest_paths_to_sensors.json", "w") as f:
    json.dump(top2_results, f, indent=4)
