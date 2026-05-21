import json, os, copy, logging
import numpy as np
import pypsa
logging.disable(logging.CRITICAL)

DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)

full = os.path.join(NETS_DIR, d['files'][0])
net = pypsa.Network(full)

print("=== Transformer list ===")
cols = [c for c in ['bus0','bus1','x','r','tap_ratio','x_pu_eff','r_pu_eff','s_nom','v_nom0','v_nom1'] if c in net.transformers.columns]
print(net.transformers[cols].to_string())
print("=== Bus v_nom ===")
print(net.buses[['v_nom']].to_string())

# Get transformer touching VL6_0
trafo_vl6 = [t for t in net.transformers.index if net.transformers.loc[t,'bus0']=='VL6_0' or net.transformers.loc[t,'bus1']=='VL6_0']
print("\nTransformers at VL6_0:", trafo_vl6)
t_row = net.transformers.loc[trafo_vl6[0]]
print(t_row.to_string())

print()
print("=== Bus data for VL6_0 and VL9_0 ===")
for b in ['VL6_0', 'VL9_0']:
    print(b, net.buses.loc[b].to_string())
    print()

# What does PyPSA actually use for transformer admittance internally?
net2 = copy.deepcopy(net)
net2.determine_network_topology()
sn_label = list(net2.sub_networks.index)[0]
sn = net2.sub_networks.at[sn_label, 'obj']
sn.calculate_Y()

# Access internal transformer admittance (PyPSA stores it in sn.transformers_t or similar)
print("=== PyPSA sn.Y full diagonal ===")
Y = np.asarray(sn.Y.todense())
buses_sn = list(sn.buses_i())
print("SN bus order:", buses_sn)
print("Y diagonal (imaginary):", np.diag(Y).imag)
print("Y diagonal (real):", np.diag(Y).real)

# Check off-diagonal entries for VL6_0 - VL9_0
all_buses = list(net.buses.index)
bus_idx = {b: i for i, b in enumerate(all_buses)}
# Find VL6 and VL9 in sn bus order
sn_idx = {b: i for i, b in enumerate(buses_sn)}
i6 = sn_idx.get('VL6_0', -1)
i9 = sn_idx.get('VL9_0', -1)
print(f"\nPyPSA Y[VL6_0, VL9_0] = {Y[i6, i9]}")
print(f"PyPSA Y[VL6_0, VL6_0] = {Y[i6, i6]}")
print(f"PyPSA Y[VL9_0, VL9_0] = {Y[i9, i9]}")

# Compare: what does manual compute?
def y_trafo_manual(row):
    if 'x_pu_eff' in row.index and not (row['x_pu_eff'] != row['x_pu_eff']) and float(row['x_pu_eff']) != 0:
        x = float(row['x_pu_eff'])
        r = float(row['r_pu_eff']) if 'r_pu_eff' in row.index else 0.0
    else:
        x = float(row['x'])
        r = float(row.get('r', 0.0))
    tap = float(row['tap_ratio']) if 'tap_ratio' in row.index else 1.0
    z = complex(r, x)
    y_se = 1.0/z if abs(z) > 1e-12 else 0.0
    tc = complex(tap, 0.)
    return y_se, tc

t_row = net.transformers.loc['T-6-9-1']
y_se, tc = y_trafo_manual(t_row)
print(f"\nManual T-6-9-1: y_series = {y_se:.4f}  tap = {tc}")
print(f"Manual Y[VL6_0, VL6_0] contribution: {y_se/(tc*tc.conjugate()):.4f}")
print(f"Manual Y[VL9_0, VL9_0] contribution: {y_se:.4f}")
print(f"Manual Y[VL6_0, VL9_0] contribution: {-y_se/tc.conjugate():.4f}")
