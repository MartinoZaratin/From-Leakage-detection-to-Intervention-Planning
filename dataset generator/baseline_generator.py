# %%
import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load configuration from YAML
# -----------------------------
sensor_nodes = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
                      'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
                      'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
                      'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                      'n726', 'n740', 'n752', 'n769']
inp_file = "L-TOWN_Real.inp"
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

# -----------------------------
# 5. Extract pressure data
# -----------------------------
pressure_df = results.node['pressure']

# Filter only the specified sensor nodes (if they exist)
valid_sensors = [n for n in sensor_nodes if n in pressure_df.columns]
missing_sensors = [n for n in sensor_nodes if n not in pressure_df.columns]

if missing_sensors:
    print(f"Warning: {len(missing_sensors)} sensors not found in the network: {missing_sensors}")

sensor_pressures = pressure_df[valid_sensors]

# -----------------------------
# 6. Add measurement noise (optional)
# -----------------------------
if noise_std > 0:
    noisy_pressures = sensor_pressures * (1 + np.random.normal(0, noise_std, sensor_pressures.shape))
else:
    noisy_pressures = sensor_pressures.copy()

# -----------------------------
# 7. Save results to CSV
# -----------------------------
noisy_pressures.to_csv(output_csv)
print(f"Saved synthetic pressure data to {output_csv}")

# -----------------------------
# 8. Plot a preview
# -----------------------------
plt.figure(figsize=(10,5))
noisy_pressures.iloc[:168].plot(ax=plt.gca())  # first week preview
plt.xlabel("Time step (hours)")
plt.ylabel("Pressure (m)")
plt.title("Simulated Pressures (first week)")
plt.legend(title='Sensor Node')
plt.tight_layout()
plt.show()

#%%
# read Baseline.csv
df = pd.read_csv('Baseline.csv', index_col=0)
# plot all  measurements for first few columns
sensor_cols = df.columns.tolist()
for i in range(len(sensor_cols)):
    plt.plot(df[sensor_cols[i]], label=sensor_cols[i])
plt.xlabel('Time point')
plt.ylabel('Pressure')
plt.title('Pressure values for all sensors')
plt.legend()
plt.grid(True)
plt.show()
# %%
# plot first sensor measurements
plt.plot(df[sensor_cols[0]], label=sensor_cols[0])
plt.xlabel('Time point')
plt.ylabel('Pressure')
plt.title('Pressure values for first sensor')
plt.legend()
plt.grid(True)
plt.show()
# %%
#####################################################
# Flow analysis for constructing the directed graph
#####################################################

flow = results.link["flowrate"]
time_step = 3600
flow_at_t = flow.loc[time_step]

# determine flow directions
flow_directions = {}
for pipe_name, q in flow_at_t.items():
    start_node = wn.get_link(pipe_name).start_node_name
    end_node = wn.get_link(pipe_name).end_node_name

    if q >= 0:
        flow_directions[pipe_name] = (start_node, end_node)
    elif q < 0:
        flow_directions[pipe_name] = (end_node, start_node)

# %%
# save flow directions to csv
flow_dir_df = pd.DataFrame.from_dict(flow_directions, orient='index', columns=['From_Node', 'To_Node'])
flow_dir_df.to_csv('flow_directions.csv')
