"""Find exact x_pu source used by calculate_Y and test with x_pu_eff"""
import json, os, copy, logging, numpy as np, pypsa, inspect
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))

# What happens to x_pu after determine_network_topology()?
net2 = copy.deepcopy(net0)
print("=== Before determine_network_topology ===")
print("transformers.x_pu:", net2.transformers.x_pu.tolist())
print("transformers.x_pu_eff:", net2.transformers.x_pu_eff.tolist())

net2.determine_network_topology()
print("\n=== After determine_network_topology ===")
print("transformers.x_pu:", net2.transformers.x_pu.tolist())

# Check sn.branches() after topology determination
sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
br = sn.branches()
trafo_entries = br.loc['Transformer'] if 'Transformer' in br.index.get_level_values(0) else None
print("\nsn.branches() transformer x_pu AFTER determine_topology:")
print(trafo_entries[['bus0','bus1','x_pu','x_pu_eff']].to_string())

# Now: manually patch x_pu = x_pu_eff for transformers and recompute Y
net3 = copy.deepcopy(net0)
net3.transformers['x_pu'] = net3.transformers['x_pu_eff']
net3.transformers['r_pu'] = net3.transformers.get('r_pu_eff', net3.transformers.get('r_pu', 0.0))
net3.determine_network_topology()
sn3 = net3.sub_networks.at[list(net3.sub_networks.index)[0], 'obj']
sn3.calculate_Y()
Y3_raw = np.asarray(sn3.Y.todense())
buses_o3 = list(sn3.buses_o)
# Check VL6_0 and VL9_0 entries
i6 = buses_o3.index('VL6_0'); i9 = buses_o3.index('VL9_0')
print(f"\nWith x_pu = x_pu_eff PATCHED:")
print(f"  Y[VL6_0, VL6_0] = {Y3_raw[i6,i6]:.4f}")
print(f"  Y[VL6_0, VL9_0] = {Y3_raw[i6,i9]:.4f}")
print(f"  Y[VL9_0, VL9_0] = {Y3_raw[i9,i9]:.4f}")

# Get pf solution
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]
net_buses = list(net_pf.buses.index)
vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
V = vmag * np.exp(1j * vang)
P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

# Reorder patched Y to net_buses order
pos_in_o3 = {b: j for j, b in enumerate(buses_o3)}
perm3 = [pos_in_o3[b] for b in net_buses]
Y3_correct = Y3_raw[np.ix_(perm3, perm3)]

vr, vi = V.real, V.imag
Ir = Y3_correct.real @ vr - Y3_correct.imag @ vi
Ii = Y3_correct.real @ vi + Y3_correct.imag @ vr
Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
mse = np.mean((Pc-P_inj)**2 + (Qc-Q_inj)**2)
maxP = np.abs(Pc-P_inj).max()
maxQ = np.abs(Qc-Q_inj).max()
print(f"\nKCL with patched Y (x_pu=x_pu_eff, buses_o corrected):")
print(f"  MSE={mse:.6e}  maxP={maxP:.4e}  maxQ={maxQ:.4e}")

# Also print full calculate_Y source
src = inspect.getsource(pypsa.components.SubNetwork.calculate_Y)
print("\n=== Full calculate_Y source ===")
print(src)
