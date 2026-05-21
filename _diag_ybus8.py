"""Per-bus KCL residual with correct buses_o reordering (using diag6 logic)"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))

# Correct reordering: diag6 approach
net2 = copy.deepcopy(net0)
net2.determine_network_topology()
sn = net2.sub_networks.at[list(net2.sub_networks.index)[0], 'obj']
sn_buses_i = list(sn.buses_i())
sn_buses_o = list(sn.buses_o)
sn.calculate_Y()
Y_raw = np.asarray(sn.Y.todense())

# perm_i_to_o[i] = position of buses_i[i] in buses_o  (correct approach)
pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
perm_i_to_o = [pos_in_o[sn_buses_i[i]] for i in range(len(sn_buses_i))]
Y_correct = Y_raw[np.ix_(perm_i_to_o, perm_i_to_o)]

# Run pf and extract solution
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]
net_buses = list(net_pf.buses.index)
vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
V = vmag * np.exp(1j * vang)
P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

# Also get INTERNAL Y from net_pf after pf()
sn_pf = net_pf.sub_networks.at[list(net_pf.sub_networks.index)[0], 'obj']
sn_pf_buses_o = list(sn_pf.buses_o)
Y_pf_raw = np.asarray(sn_pf.Y.todense())
pos_in_o_pf = {b: j for j, b in enumerate(sn_pf_buses_o)}
perm_pf = [pos_in_o_pf[net_buses[i]] for i in range(len(net_buses))]
Y_pf_correct = Y_pf_raw[np.ix_(perm_pf, perm_pf)]

vr, vi = V.real, V.imag
for Y, label in [(Y_correct, 'Y_calc_corrected'), (Y_pf_correct, 'Y_pf_corrected')]:
    Ir = Y.real @ vr - Y.imag @ vi
    Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
    P_err = np.abs(Pc - P_inj)
    Q_err = np.abs(Qc - Q_inj)
    mse = np.mean(P_err**2 + Q_err**2)
    print(f"\n=== {label}: MSE={mse:.6e} ===")
    print(f"{'Bus':<12} {'P_err':>10} {'Q_err':>10} {'P_inj':>8} {'Q_inj':>8}")
    for i, bus in enumerate(net_buses):
        if P_err[i] > 1e-3 or Q_err[i] > 1e-3:
            print(f"{bus:<12} {P_err[i]:10.6f} {Q_err[i]:10.6f} {P_inj[i]:8.4f} {Q_inj[i]:8.4f}")

# Are the two corrected Y matrices identical?
diff = np.max(np.abs(Y_correct - Y_pf_correct))
print(f"\nMax diff between Y_calc and Y_pf (both corrected): {diff:.6e}")

# Check: is buses_o ordering the same in both cases?
print("buses_o same?", sn_buses_o == sn_pf_buses_o)

# Last check: does Y contain the shunt B10-SH 1 (b=0.19 at VL10_0)?
i10 = net_buses.index('VL10_0')
print(f"\nY_correct diagonal at VL10_0: {Y_correct[i10,i10]:.4f}")
print(f"Expected: manual lines sum + 0.19j shunt")
