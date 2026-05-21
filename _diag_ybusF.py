"""Test actual physics loss function with perfect predictions to measure true floor"""
import sys, os, json, copy, logging, numpy as np
sys.path.insert(0, r'C:\git_repos\Graph-Neural-Networks')
logging.disable(logging.CRITICAL)

DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')

# Need to import from notebook - let's do it by running the notebook cells first
# Instead, let's replicate the physics loss function directly
import pypsa, torch

def compute_Y_corrected(net):
    """Get Y matrix correctly ordered by network.buses.index."""
    import copy, numpy as np
    net2 = copy.deepcopy(net)
    net2.determine_network_topology()
    sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
    sn_buses_i = list(sn.buses_i())
    sn_buses_o = list(sn.buses_o)
    sn.calculate_Y()  # calls calculate_dependent_values internally
    Y_raw = np.asarray(sn.Y.todense())
    if sn_buses_i == sn_buses_o:
        return Y_raw, sn_buses_i
    pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
    perm_i_to_o = [pos_in_o[b] for b in sn_buses_i]
    return Y_raw[np.ix_(perm_i_to_o, perm_i_to_o)], sn_buses_i

def kcl_residual_perfect(net, Y):
    """Compute KCL residual at TRUE PyPSA solution (perfect prediction)."""
    net_pf = copy.deepcopy(net)
    net_pf.pf(snapshots=net_pf.snapshots[0:1])
    snap = net_pf.snapshots[0]
    net_buses = list(net_pf.buses.index)
    vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
    vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
    V = vmag * np.exp(1j * vang)
    P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
    Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values
    
    vr, vi = V.real, V.imag
    Ir = Y.real @ vr - Y.imag @ vi
    Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii
    Qc = vi*Ir - vr*Ii
    
    res_P = Pc - P_inj
    res_Q = Qc - Q_inj
    return np.mean(res_P**2 + res_Q**2), np.abs(res_P).max(), np.abs(res_Q).max()

# Test on 20 mixed networks
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)

print(f"{'File (truncated)':<20} {'buses':>6} {'shunts':>6} {'MSE':>12} {'maxP':>10} {'maxQ':>10}")
print("-"*70)
results = []
for fname in d['files'][:20]:
    net0 = pypsa.Network(os.path.join(NETS_DIR, fname))
    try:
        Y, _ = compute_Y_corrected(net0)
        mse, mp, mq = kcl_residual_perfect(net0, Y)
        n_sh = len(net0.shunt_impedances)
        n_bus = len(net0.buses)
        tag = os.path.basename(fname)[-10:]
        print(f"{tag:<20} {n_bus:6d} {n_sh:6d} {mse:12.4e} {mp:10.4e} {mq:10.4e}")
        results.append(mse)
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nOverall: mean MSE={np.mean(results):.4e}  max={max(results):.4e}  min={min(results):.4e}")
