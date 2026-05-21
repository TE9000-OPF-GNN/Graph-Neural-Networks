"""Check buses_o vs buses_i for ieee9 and verify physics floor fix"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')

def kcl_mse(net, Y):
    """KCL residual at true pf solution (v_ang in RADIANS, no deg2rad)"""
    net_pf = copy.deepcopy(net)
    net_pf.pf(snapshots=net_pf.snapshots[0:1])
    snap = net_pf.snapshots[0]
    net_buses = list(net_pf.buses.index)
    vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
    vang = net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values  # NO deg2rad!
    V = vmag * np.exp(1j * vang)
    P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
    Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values
    vr, vi = V.real, V.imag
    Ir = Y.real @ vr - Y.imag @ vi; Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
    return np.mean((Pc-P_inj)**2 + (Qc-Q_inj)**2)

def get_Y_buggy(net):
    """Current (buggy) implementation: Y in buses_o order returned as-is"""
    net2 = copy.deepcopy(net)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    sn_bus_order = list(sn.buses_i())
    net_bus_order = list(range(len(net2.buses)))
    sn.calculate_Y()
    Y_dense = np.asarray(sn.Y.todense())
    if sorted(sn_bus_order) == net_bus_order:  # always False (string vs int)
        return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]
    else:
        return Y_dense  # Y in buses_o order!

def get_Y_fixed(net):
    """Fixed: correctly reorder from buses_o to buses_i"""
    net2 = copy.deepcopy(net)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    sn_buses_i = list(sn.buses_i())
    sn_buses_o = list(sn.buses_o)
    sn.calculate_Y()
    Y_raw = np.asarray(sn.Y.todense())
    if sn_buses_i == sn_buses_o:
        return Y_raw  # same order, no reordering needed
    pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
    perm = [pos_in_o[b] for b in sn_buses_i]
    return Y_raw[np.ix_(perm, perm)]

# Test on mixed networks and ieee9
print("=== Mixed networks (ieee30/cigre14 variants) ===")
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)

for fname in d['files'][:5]:
    net0 = pypsa.Network(os.path.join(NETS_DIR, fname))
    # Check ordering
    net2 = copy.deepcopy(net0)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    same_order = list(sn.buses_i()) == list(sn.buses_o)
    
    Y_bug = get_Y_buggy(net0)
    Y_fix = get_Y_fixed(net0)
    mse_bug = kcl_mse(net0, Y_bug)
    mse_fix = kcl_mse(net0, Y_fix)
    n_bus = len(net0.buses)
    tag = os.path.basename(fname)[-8:]
    print(f"  {tag} buses={n_bus:2d} buses_i==buses_o={same_order}  MSE_buggy={mse_bug:.2e}  MSE_fixed={mse_fix:.2e}")

# Test on ieee9-like networks from the mixed dataset (9-bus entries)
print("\n=== 9-bus networks from mixed_1500_wide ===")
count = 0
for fname in d['files']:
    if count >= 5: break
    net0 = pypsa.Network(os.path.join(NETS_DIR, fname))
    if len(net0.buses) != 9: continue
    count += 1
    net2 = copy.deepcopy(net0)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    same_order = list(sn.buses_i()) == list(sn.buses_o)
    
    Y_bug = get_Y_buggy(net0)
    Y_fix = get_Y_fixed(net0)
    mse_bug = kcl_mse(net0, Y_bug)
    mse_fix = kcl_mse(net0, Y_fix)
    tag = os.path.basename(fname)[-8:]
    print(f"  {tag} buses=9 buses_i==buses_o={same_order}  MSE_buggy={mse_bug:.2e}  MSE_fixed={mse_fix:.2e}")
