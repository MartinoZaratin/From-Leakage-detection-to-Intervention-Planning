# %%
import wntr
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load configuration from YAML
# -----------------------------
yaml_file = 'dataset_configuration.yaml'  # your YAML file
with open(yaml_file, 'r') as f:
    config = yaml.safe_load(f)

sensor_nodes = config.get('pressure_sensors', [])
inp_file = "L-TOWN_Real.inp"
output_csv = config.get('output_csv', 'Baseline.csv')
duration_days = config.get('duration_days', 365)
report_step_min = config.get('report_step_hr', 15)  # in minutes
noise_std = config.get('noise_std', 0.0)

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
