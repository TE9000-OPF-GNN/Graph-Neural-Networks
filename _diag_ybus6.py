"""Prove Y matrix is in buses_o order and fix the reordering"""
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
sn.calculate_Y()

sn_buses_i = list(sn.buses_i())
sn_buses_o = list(sn.buses_o)
print("buses_i order (first 15):", sn_buses_i[:15])
print()
print("buses_o order (first 15):", sn_buses_o[:15])
print()
print("Same order?", sn_buses_i == sn_buses_o)

# Find positions of VL6_0 and VL9_0 in buses_o
i6_o = sn_buses_o.index('VL6_0')
i9_o = sn_buses_o.index('VL9_0')
i6_i = sn_buses_i.index('VL6_0')
i9_i = sn_buses_i.index('VL9_0')
print(f"\nVL6_0: buses_i pos={i6_i}, buses_o pos={i6_o}")
print(f"VL9_0: buses_i pos={i9_i}, buses_o pos={i9_o}")

Y_raw = np.asarray(sn.Y.todense())
print(f"\nAssuming buses_o indexing:")
print(f"  Y[VL6_0, VL6_0] = {Y_raw[i6_o, i6_o]:.4f}")
print(f"  Y[VL9_0, VL9_0] = {Y_raw[i9_o, i9_o]:.4f}")
print(f"  Y[VL6_0, VL9_0] = {Y_raw[i6_o, i9_o]:.4f}")

print(f"\nAssuming buses_i indexing (WRONG):")
print(f"  Y[VL6_0, VL6_0] = {Y_raw[i6_i, i6_i]:.4f}")
print(f"  Y[VL9_0, VL9_0] = {Y_raw[i9_i, i9_i]:.4f}")
print(f"  Y[VL6_0, VL9_0] = {Y_raw[i6_i, i9_i]:.4f}")

# Reorder Y to match buses_i (= net_buses) order using buses_o
# buses_o -> buses_i reordering
pos_in_i = {b: i for i, b in enumerate(sn_buses_i)}
perm_o_to_i = [pos_in_i[b] for b in sn_buses_o]  # perm_o_to_i[j] = position of buses_o[j] in buses_i
# To get Y in buses_i order: Y_correct[i,j] = Y_raw[inv(perm)[i], inv(perm)[j]]
# where perm = [buses_o position of buses_i[k] for k in range(N)]
inv_perm = [0]*len(sn_buses_i)
for j, b in enumerate(sn_buses_o):
    i = pos_in_i[b]
    inv_perm[i] = j
Y_correct = Y_raw[np.ix_(inv_perm, inv_perm)]

print(f"\nWith correct reordering (buses_o -> buses_i):")
print(f"  Y[VL6_0, VL6_0] = {Y_correct[i6_i, i6_i]:.4f}")
print(f"  Y[VL9_0, VL9_0] = {Y_correct[i9_i, i9_i]:.4f}")
print(f"  Y[VL6_0, VL9_0] = {Y_correct[i6_i, i9_i]:.4f}")

# Now compute KCL residual with corrected Y
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]
net_buses = list(net_pf.buses.index)

vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
V = vmag * np.exp(1j * vang)
P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

def kcl(V, Y, P, Q):
    vr, vi = V.real, V.imag
    Ir = Y.real @ vr - Y.imag @ vi
    Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
    return np.mean((Pc-P)**2+(Qc-Q)**2), np.abs(Pc-P).max(), np.abs(Qc-Q).max()

mse_wrong, _, _ = kcl(V, Y_raw, P_inj, Q_inj)
mse_correct, mp, mq = kcl(V, Y_correct, P_inj, Q_inj)
print(f"\nKCL MSE with buses_i (wrong) order: {mse_wrong:.6e}")
print(f"KCL MSE with buses_o corrected:     {mse_correct:.6e}  maxP={mp:.4e}  maxQ={mq:.4e}")
