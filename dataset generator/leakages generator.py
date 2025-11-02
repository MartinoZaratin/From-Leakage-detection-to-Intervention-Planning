#%%
import wntr
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
#%%


def leak_diameter_to_emitter(D_leak_m, C_d=0.75):
    """Convert leak diameter (m) to emitter coefficient (m²/√m)."""
    g = 9.81
    A = math.pi * (D_leak_m / 2) ** 2
    C = C_d * A * math.sqrt(2 * g)
    return C


def generate_leakage_data(inp_file, leak_node, leak_diam_mm, duration_days, report_step_min, sensor_nodes):
    # -----------------------------
    # 1. Load the network
    # -----------------------------
    wn = wntr.network.WaterNetworkModel(inp_file)

    # -----------------------------
    # 2. Add leak as emitter at specified node
    # -----------------------------
    if leak_node in wn.node_name_list:
        node = wn.get_node(leak_node)
        emitter_C = leak_diameter_to_emitter(leak_diam_mm / 1000)
        node.emitter_coefficient = emitter_C
        print(f"Added background leak at node '{leak_node}' "
            f"(diameter = {leak_diam_mm} mm, emitter = {emitter_C:.6f} m²/√m)")
    else:
        print(f" Warning: node '{leak_node}' not found in the network. No leak added.")

    # -----------------------------
    # 3. Simulation settings
    # -----------------------------
    wn.options.time.duration = duration_days * 24 * 3600       # total simulation time [s]
    wn.options.time.hydraulic_timestep = report_step_min * 60  # [s]
    wn.options.time.report_timestep = report_step_min * 60     # [s]
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

    valid_sensors = [n for n in sensor_nodes if n in pressure_df.columns]

    sensor_pressures = pressure_df[valid_sensors]

    # -----------------------------
    # 6. Save results to CSV
    # -----------------------------
    output_csv = f'Datasets\Leakages {leak_diam_mm}mm\leakage_{leak_node}_{leak_diam_mm}.csv'
    sensor_pressures.to_csv(output_csv)
    sensor_pressures.index.name = 'Time (s)'
    print(f"Saved synthetic pressure data to {output_csv}")



    
# -----------------------------
# Set configuration
# -----------------------------
sensor_nodes = ['n1', 'n4', 'n31', 'n54', 'n105', 'n114', 'n163', 'n188',
                      'n215', 'n229', 'n288', 'n296', 'n332', 'n342', 'n410',
                      'n415', 'n429', 'n458', 'n469', 'n495', 'n506', 'n516',
                      'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                      'n726', 'n740', 'n752', 'n769']
inp_file = "L-TOWN_Real.inp"
duration_days = 90
report_step_min = 15  # in minutes

# leak parameters
leak_diam_mm = 15  # leak diameter in mm

# 389 leak nodes
leak_nodes = ['n3','n4','n5','n6','n10','n11','n14','n15','n17','n18','n22','n23',
 'n25','n26','n32','n33','n34','n37','n38','n40','n41','n43','n44','n46',
 'n48','n50','n51','n52','n58','n61','n63','n64','n67','n68','n71','n72',
 'n74','n75','n77','n78','n79','n82','n83','n86','n87','n91','n96','n97',
 'n98','n100','n101','n104','n105','n106','n107','n111','n112','n113','n114','n115',
 'n117','n121','n122','n125','n128','n129','n131','n132','n133','n135','n137','n140',
 'n142','n143','n144','n147','n149','n151','n152','n153','n156','n157','n159','n163',
 'n164','n165','n166','n169','n171','n172','n174','n177','n180','n182','n183','n187',
 'n188','n189','n191','n192','n193','n194','n198','n199','n201','n203','n205','n209',
 'n212','n213','n217','n219','n222','n223','n227','n229','n232','n233','n234','n238',
 'n239','n242','n243','n244','n245','n248','n253','n255','n256','n257','n258','n261',
 'n262','n263','n264','n265','n268','n269','n271','n275','n276','n277','n279','n281',
 'n283','n285','n286','n287','n288','n290','n292','n294','n298','n299','n300','n302',
 'n303','n306','n307','n309','n310','n311','n313','n314','n315','n320','n321','n323',
 'n325','n327','n329','n330','n331','n333','n335','n336','n338','n339','n340','n345',
 'n346','n347','n349','n351','n354','n356','n358','n360','n363','n366','n371','n372',
 'n373','n375','n378','n380','n381','n383','n388','n390','n392','n394','n396','n398',
 'n401','n403','n405','n408','n409','n410','n412','n414','n416','n418','n421','n423',
 'n426','n427','n429','n431','n433','n435','n439','n441','n444','n445','n446','n448',
 'n451','n452','n455','n457','n459','n461','n463','n465','n466','n469','n471','n476',
 'n478','n480','n485','n486','n487','n489','n493','n497','n499','n500','n502','n505',
 'n507','n509','n510','n513','n514','n516','n518','n521','n522','n525','n526','n529',
 'n534','n536','n537','n538','n541','n542','n545','n547','n549','n552','n553','n555',
 'n556','n557','n559','n561','n564','n566','n568','n570','n572','n573','n575','n577',
 'n578','n580','n582','n584','n587','n590','n591','n592','n594','n596','n598','n600',
 'n603','n605','n607','n609','n611','n613','n614','n615','n617','n619','n621','n622',
 'n623','n624','n625','n626','n627','n628','n632','n634','n636','n638','n640','n643',
 'n646','n648','n649','n651','n652','n653','n655','n657','n658','n659','n661','n663',
 'n665','n667','n669','n671','n673','n675','n676','n680','n682','n685','n687','n688',
 'n690','n691','n693','n695','n697','n699','n701','n704','n709','n711','n714','n716',
 'n719','n721','n722','n725','n726','n727','n729','n731','n734','n735','n737','n739',
 'n742','n744','n747','n749','n751','n753','n755','n757','n760','n763','n765','n767',
 'n769','n772','n779','n781','n782']



# run simulation and generate data
start_time = time.time()
for leak_node in leak_nodes:
    generate_leakage_data(inp_file, leak_node, leak_diam_mm, duration_days, report_step_min, sensor_nodes)
end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
