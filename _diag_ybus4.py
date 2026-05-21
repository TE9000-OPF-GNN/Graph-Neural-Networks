"""Diagnose VL11_0 large admittance and find what Y PyPSA's pf() actually uses"""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))

# ---------- all lines at VL11_0 ----------
print("=== Lines at VL11_0 ===")
for l in net0.lines.index:
    row = net0.lines.loc[l]
    if 'VL11_0' in [row.bus0, row.bus1]:
        print(f"  {l}: {row.bus0}-{row.bus1}  r={row.r:.6f}  x={row.x:.6f}  b={row.get('b',0):.4f}")

print("=== Shunt impedances ===")
if len(net0.shunt_impedances) > 0:
    print(net0.shunt_impedances[['bus','b','g']].to_string())
else:
    print("  (none)")

# ---------- what does pf() use internally? ----------
# After pf(), extract the sub-network's Y and test KCL using pf's own V and S
net_pf = copy.deepcopy(net0)
net_pf.pf(snapshots=net_pf.snapshots[0:1])
snap = net_pf.snapshots[0]
net_buses = list(net_pf.buses.index)
sn = net_pf.sub_networks.at[list(net_pf.sub_networks.index)[0], 'obj']
sn_buses = list(sn.buses_i())
Y_internal = np.asarray(sn.Y.todense())
# Reorder to net order
pos = {b: i for i, b in enumerate(sn_buses)}
perm = [pos[b] for b in net_buses]
Y_int = Y_internal[np.ix_(perm, perm)]

vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
V = vmag * np.exp(1j * vang)
P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

# KCL with internal Y
vr, vi = V.real, V.imag
Ir = Y_int.real @ vr - Y_int.imag @ vi
Ii = Y_int.real @ vi + Y_int.imag @ vr
Pc = vr*Ir + vi*Ii
Qc = vi*Ir - vr*Ii
P_err = np.abs(Pc - P_inj)
Q_err = np.abs(Qc - Q_inj)

print("\n=== Per-bus KCL errors with PyPSA internal Y ===")
print(f"{'Bus':<12} {'P_err':>12} {'Q_err':>12} {'P_inj':>10} {'Q_inj':>10} {'Pc':>10} {'Qc':>10}")
for i, bus in enumerate(net_buses):
    if P_err[i] > 0.001 or Q_err[i] > 0.001:
        print(f"{bus:<12} {P_err[i]:12.6f} {Q_err[i]:12.6f} {P_inj[i]:10.4f} {Q_inj[i]:10.4f} {Pc[i]:10.4f} {Qc[i]:10.4f}")

# Also try using V directly from NR internal (different convention?)
# PyPSA stores V after convergence in sn.v_sol or sn.buses_t.v_mag_pu?
# Check: what is pf()'s actual S_inj computation?
# PyPSA pf uses: S = V * conj(I) = V * conj(Y @ V)
# where V is complex voltage, Y is sub_network.Y
# Let me compute it directly the PyPSA way
V_sn = np.array([vmag[perm[pos[b]]] * np.exp(1j * vang[perm[pos[b]]]) for b in sn_buses])
# Wait, perm maps net_buses->sn_buses: perm[i] = position of net_buses[i] in sn_buses
# So V in sn order:
inv_perm = [0]*len(perm)
for i, p in enumerate(perm): inv_perm[p] = i
V_sn_order = np.array([V[inv_perm[k]] for k in range(len(sn_buses))])
S_pypsa_way = V_sn_order * np.conj(Y_internal @ V_sn_order)
P_pypsa = S_pypsa_way.real
Q_pypsa = S_pypsa_way.imag
# Map back to net order
P_reord = np.array([P_pypsa[pos[b]] for b in net_buses])
Q_reord = np.array([Q_pypsa[pos[b]] for b in net_buses])
mse_pypsa = np.mean((P_reord - P_inj)**2 + (Q_reord - Q_inj)**2)
print(f"\nPyPSA-style KCL (S = V*conj(Y@V)): MSE={mse_pypsa:.6e}")
worst = np.argmax(np.abs(P_reord - P_inj) + np.abs(Q_reord - Q_inj))
print(f"  Worst: {net_buses[worst]}  P_err={abs(P_reord[worst]-P_inj[worst]):.4e}  Q_err={abs(Q_reord[worst]-Q_inj[worst]):.4e}")
