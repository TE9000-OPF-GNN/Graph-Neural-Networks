"""KCL check in PyPSA's native buses_o ordering to verify NR convergence"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]

# Get sn Y and buses_o from POST-pf network
sn = net_pf.sub_networks.at[list(net_pf.sub_networks.index)[0], 'obj']
buses_o = list(sn.buses_o)
Y_raw = np.asarray(sn.Y.todense())  # in buses_o ordering

# Get voltage in buses_o ordering
vmag_o = net_pf.buses_t.v_mag_pu.loc[snap].reindex(buses_o).values
vang_o = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(buses_o).values)
V_o = vmag_o * np.exp(1j * vang_o)

# S_calc in buses_o order
S_calc_o = V_o * np.conj(Y_raw @ V_o)
P_calc_o = S_calc_o.real
Q_calc_o = S_calc_o.imag

# P_inj in buses_o order
P_inj_o = net_pf.buses_t.p.loc[snap].reindex(buses_o).values
Q_inj_o = net_pf.buses_t.q.loc[snap].reindex(buses_o).values

print("=== KCL in buses_o order (native PyPSA ordering) ===")
P_err = np.abs(P_calc_o - P_inj_o)
Q_err = np.abs(Q_calc_o - Q_inj_o)
mse = np.mean(P_err**2 + Q_err**2)
print(f"MSE: {mse:.6e}  maxP_err: {P_err.max():.4e}  maxQ_err: {Q_err.max():.4e}")
print()
print(f"{'Bus':<12} {'P_err':>10} {'Q_err':>10} {'P_inj':>8} {'Q_inj':>8} {'P_calc':>8} {'Q_calc':>8}")
for i, bus in enumerate(buses_o):
    if P_err[i] > 0.001 or Q_err[i] > 0.001:
        print(f"{bus:<12} {P_err[i]:10.6f} {Q_err[i]:10.6f} {P_inj_o[i]:8.4f} {Q_inj_o[i]:8.4f} {P_calc_o[i]:8.4f} {Q_calc_o[i]:8.4f}")

print()
# Also get the x_pu used in calculate_Y by inspecting the full source
import inspect, pypsa.components as pc
src = inspect.getsource(pc.SubNetwork.calculate_Y)
# Look for x_pu_eff usage
for i, line in enumerate(src.split('\n')):
    if 'x_pu' in line or 'r_pu' in line or 'zero' in line or 'nan' in line.lower() or 'replace' in line.lower():
        print(f"L{i:3d}: {line.rstrip()}")
