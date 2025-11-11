
import geopandas as gpd
import networkx as nx
import json

# Load shapefiles
nodes = gpd.read_file("..\dataset generator\_shapefiles_\L-TOWN_Real_junctions.shp")
pipes = gpd.read_file("..\dataset generator\_shapefiles_\L-TOWN_Real_pipes.shp")

# not completely right: missing pump and valves connections

# Build graph
G = nx.Graph()
for _, row in pipes.iterrows():
    G.add_edge(row["nodefrom"], row["nodeto"], weight=row["length"])

# Define sensors (subset of node IDs)
sensors = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
            'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
            'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
            'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
            'n726', 'n740', 'n752', 'n769']

# Compute distances and paths
lengths, paths = nx.multi_source_dijkstra(G, sources=sensors, weight="weight")   # lengths and paths are dicts

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

combined_results = {
    node: {
        "closest_sensor": closest_sensors[node],
        "distance": distances[node]
    }
    for node in G.nodes()
}

# Save results to JSON
with open("shortest_paths_to_sensors.json", "w") as f:
    json.dump(combined_results, f, indent=4)


# add information to node shapefiles
save_to_shapefile = False
if save_to_shapefile:
    nodes["closest_sensor"] = nodes["id"].map(closest_sensors)
    nodes["distance_to_sensor"] = nodes["id"].map(distances)
    nodes.to_file("L-TOWN_Real_junctions_with_sensor_info.shp")
