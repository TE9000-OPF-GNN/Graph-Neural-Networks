"""Inspect full Y matrix rows for VL9_0 and VL11_0"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))
net2 = copy.deepcopy(net0)
net2.determine_network_topology()
sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
sn_buses = list(sn.buses_i())
sn.calculate_Y()
Y = np.asarray(sn.Y.todense())
btoi = {b: i for i, b in enumerate(sn_buses)}

# Print nonzero entries in row for VL9_0 and VL11_0
print("Nonzero Y entries for VL9_0 row (all buses with |Y|>0.001):")
i9 = btoi['VL9_0']
for j, bus in enumerate(sn_buses):
    v = Y[i9, j]
    if abs(v) > 0.001:
        print(f"  Y[VL9_0, {bus}] = {v:.4f}")

print()
print("Nonzero Y entries for VL11_0 row:")
i11 = btoi['VL11_0']
for j, bus in enumerate(sn_buses):
    v = Y[i11, j]
    if abs(v) > 0.001:
        print(f"  Y[VL11_0, {bus}] = {v:.4f}")

print()
print("Nonzero Y entries for VL6_0 row:")
i6 = btoi['VL6_0']
for j, bus in enumerate(sn_buses):
    v = Y[i6, j]
    if abs(v) > 0.001:
        print(f"  Y[VL6_0, {bus}] = {v:.4f}")

# Also check if Y is symmetric
print()
asymm = np.max(np.abs(Y - Y.T))
print(f"Max asymmetry |Y - Y^T|: {asymm:.6e}")

# Check what P_inj physically represents: generator minus load at each bus
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]
print()
print("=== Bus injections vs Y*V calculation for a few buses ===")
# At VL9_0: what SHOULD Y*V give?
# VL9_0 has only lines, no generator, no load → P_inj should be 0
print(f"VL9_0: P_inj={net_pf.buses_t.p.loc[snap,'VL9_0']:.6f}  Q_inj={net_pf.buses_t.q.loc[snap,'VL9_0']:.6f}")
print(f"VL11_0: P_inj={net_pf.buses_t.p.loc[snap,'VL11_0']:.6f}  Q_inj={net_pf.buses_t.q.loc[snap,'VL11_0']:.6f}")
print(f"VL6_0: P_inj={net_pf.buses_t.p.loc[snap,'VL6_0']:.6f}  Q_inj={net_pf.buses_t.q.loc[snap,'VL6_0']:.6f}")

# Let's look at the subnetwork buses_o array (bus ordering within pf)
print()
print("sn buses_o (first 5):", sn.buses_o[:5] if hasattr(sn,'buses_o') else 'N/A')
