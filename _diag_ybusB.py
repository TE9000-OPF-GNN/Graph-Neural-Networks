"""Check what x_pu values are in sn.branches() and how PyPSA handles p_set"""
import json, os, copy, logging, numpy as np, pypsa, pandas as pd
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))
net2 = copy.deepcopy(net0)
net2.determine_network_topology()
sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']

# Inspect branches() 
branches = sn.branches()
print("=== sn.branches() columns ===")
print(list(branches.columns))
print()
print("=== Transformer entries in branches ===")
trafo_rows = branches.xs('Transformer', level=0) if 'Transformer' in branches.index.get_level_values(0) else None
if trafo_rows is not None:
    print(trafo_rows[['bus0','bus1','r_pu','x_pu','x_pu_eff','tap_ratio','tap_side']].to_string())
else:
    print("No Transformer level in branches index")
    print("Index levels:", branches.index.names)
    print(branches.head(5)[['bus0','bus1','r_pu','x_pu']].to_string())

# What does sn.buses_o look like? This is used for bus indices in Y
print("\n=== sn.buses_o ===")
print(list(sn.buses_o))

# Check the actual p_set for each network component (what does PyPSA actually solve for?)
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]

print("\n=== Generator p values after pf() ===")
print(net_pf.generators_t.p.loc[snap].to_string())
print("\n=== Load p_set values ===")
print(net_pf.loads.p_set.to_string())

# What is buses_t.p exactly? Is it = P_gen - P_load?
net_buses = list(net_pf.buses.index)
buses_t_p = net_pf.buses_t.p.loc[snap].reindex(net_buses)
# Compute gen-load manually
gen_p = pd.Series(0.0, index=net_buses)
for g in net_pf.generators.index:
    bus = net_pf.generators.loc[g, 'bus']
    if g in net_pf.generators_t.p.columns:
        gen_p[bus] += net_pf.generators_t.p.loc[snap, g]

load_p = pd.Series(0.0, index=net_buses)
for l in net_pf.loads.index:
    bus = net_pf.loads.loc[l, 'bus']
    p_set = net_pf.loads_t.p_set.loc[snap, l] if l in net_pf.loads_t.p_set.columns else net_pf.loads.loc[l, 'p_set']
    load_p[bus] += p_set

net_inj = gen_p - load_p
print("\n=== Comparison: buses_t.p vs gen-load ===")
diff = np.max(np.abs(buses_t_p - net_inj))
print(f"Max |buses_t.p - (gen-load)|: {diff:.6e}")
nonzero = (buses_t_p - net_inj).abs() > 0.001
if nonzero.any():
    print("Discrepant buses:")
    print(pd.DataFrame({'buses_t.p': buses_t_p[nonzero], 'gen-load': net_inj[nonzero], 'diff': (buses_t_p-net_inj)[nonzero]}))
