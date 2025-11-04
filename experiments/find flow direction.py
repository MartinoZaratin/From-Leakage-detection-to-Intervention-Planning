import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# 1. RUN SIMULATION
# -----------------------------
# 1. Load configuration from YAML
# -----------------------------
sensor_nodes = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
                      'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
                      'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
                      'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                      'n726', 'n740', 'n752', 'n769']
inp_file = "..\\dataset generator\\L-TOWN_Real.inp"
output_csv = 'Baseline.csv'
duration_days = 365
report_step_min = 15  # in minutes
noise_std = 0.0

print(f"Loaded {len(sensor_nodes)} pressure sensors from YAML.")

# -----------------------------
# 2. Load the network
# -----------------------------
wn = wntr.network.WaterNetworkModel(inp_file)

# -----------------------------
# 3. Simulation settings
# -----------------------------
wn.options.time.duration = duration_days * 24 * 3600       # total simulation time [s]
wn.options.time.hydraulic_timestep = report_step_min * 60  # [s]
wn.options.time.report_timestep = report_step_min * 60     # [s]

# Optional: ensure demand-driven analysis
wn.options.hydraulic.demand_model = 'PDD'  # Pressure Dependent Demand

# -----------------------------
# 4. Run hydraulic simulation
# -----------------------------
print("Running hydraulic simulation...")
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()


# 2. FIND FLOW DIRECTIONS
# -----------------------------
flow_df = results.link['flowrate']
flow_directions = {}
for link_name in flow_df.columns:
    flow_series = flow_df[link_name]
    avg_flow = flow_series.mean()
    if avg_flow > 0:
        flow_directions[link_name] = 'from start node to end node'
    elif avg_flow < 0:
        flow_directions[link_name] = 'from end node to start node'
    else:
        flow_directions[link_name] = 'no flow'


# load pipe shape file
pipes = gpd.read_file("..\_shapefiles_\\L-TOWN_Real_pipes.shp")

# if geometry is the wrong direction, reverse it
for idx, row in pipes.iterrows():
    link_name = row['id']
    if link_name in flow_directions:
        direction = flow_directions[link_name]
        if direction == 'from end node to start node':
            # reverse the geometry
            pipes.at[idx, 'geometry'] = row['geometry'].reverse()

# save updated shapefile
pipes.to_file("..\_shapefiles_\\pipes_with_flow_directions_encoded_in_geometry.shp")