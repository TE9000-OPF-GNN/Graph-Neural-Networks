"""Full validation: corrected Y + shunts vs NR solution, multiple networks"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')

def get_Y_corrected(net):
    """Get PyPSA Y correctly ordered by network.buses.index (= buses_i order)."""
    net2 = copy.deepcopy(net)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    sn_buses_i = list(sn.buses_i())
    sn_buses_o = list(sn.buses_o)
    sn.calculate_Y()
    Y_raw = np.asarray(sn.Y.todense())
    # Y_raw is in buses_o order; reorder to buses_i order
    if sn_buses_i == sn_buses_o:
        return Y_raw  # already same order
    pos_in_i = {b: i for i, b in enumerate(sn_buses_i)}
    inv_perm = [pos_in_i[sn_buses_o[j]] for j in range(len(sn_buses_o))]
    return Y_raw[np.ix_(inv_perm, inv_perm)]

def kcl_residual(V, Y, P, Q):
    vr, vi = V.real, V.imag
    Ir = Y.real @ vr - Y.imag @ vi
    Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
    return np.mean((Pc-P)**2+(Qc-Q)**2)

print("=== Testing corrected Y on 5 mixed networks ===")
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)

for fname in d['files'][:10]:
    net0 = pypsa.Network(os.path.join(NETS_DIR, fname))
    net_pf = copy.deepcopy(net0)
    net_pf.pf(snapshots=net_pf.snapshots[0:1])
    snap = net_pf.snapshots[0]
    net_buses = list(net_pf.buses.index)
    vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
    vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
    V = vmag * np.exp(1j * vang)
    P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
    Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

    Y_corrected = get_Y_corrected(net0)
    mse = kcl_residual(V, Y_corrected, P_inj, Q_inj)
    n_bus = len(net_buses)
    n_sh = len(net0.shunt_impedances)
    tag = os.path.basename(fname)[:30]
    print(f"  {tag:32s} buses={n_bus:3d} shunts={n_sh} MSE={mse:.4e}")

# Now test on ieee9 networks
print()
with open(os.path.join(NETS_DIR, 'networks_ieee9_large_singel_slack.json')) as f:
    d9 = json.load(f)
print("=== IEEE9 networks ===")
for fname in d9['files'][:5]:
    net0 = pypsa.Network(os.path.join(NETS_DIR, fname))
    net_pf = copy.deepcopy(net0)
    net_pf.pf(snapshots=net_pf.snapshots[0:1])
    snap = net_pf.snapshots[0]
    net_buses = list(net_pf.buses.index)
    vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
    vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
    V = vmag * np.exp(1j * vang)
    P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
    Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values
    Y_corrected = get_Y_corrected(net0)
    mse = kcl_residual(V, Y_corrected, P_inj, Q_inj)
    tag = os.path.basename(fname)[:30]
    print(f"  {tag:32s} buses={len(net_buses):2d} MSE={mse:.4e}")
