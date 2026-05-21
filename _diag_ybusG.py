"""Verify v_ang units and check Q_calc at PV buses vs buses_t.q"""
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
net_buses = list(net_pf.buses.index)

# Print raw v_ang values
print("=== Raw v_ang values from buses_t.v_ang ===")
v_ang = net_pf.buses_t.v_ang.loc[snap].reindex(net_buses)
print(v_ang.to_string())
print(f"\nMin ang: {v_ang.min():.4f}  Max ang: {v_ang.max():.4f}")
print("(If in degrees: expect range ~[-30, +5])")
print("(If in radians: expect range ~[-0.5, +0.1])")

# Try both: v_ang as degrees vs radians
vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang_deg = np.deg2rad(v_ang.values)
vang_rad = v_ang.values  # if already radians

Y_raw_pf = np.asarray(net_pf.sub_networks.at[list(net_pf.sub_networks.index)[0], 'obj'].Y.todense())
buses_o = list(net_pf.sub_networks.at[list(net_pf.sub_networks.index)[0], 'obj'].buses_o)
pos_o = {b: j for j, b in enumerate(buses_o)}
perm = [pos_o[b] for b in net_buses]
Y = Y_raw_pf[np.ix_(perm, perm)]

P_inj = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

def kcl_mse(vmag, vang, Y, P, Q):
    V = vmag * np.exp(1j * vang)
    vr, vi = V.real, V.imag
    Ir = Y.real @ vr - Y.imag @ vi; Ii = Y.real @ vi + Y.imag @ vr
    Pc = vr*Ir + vi*Ii; Qc = vi*Ir - vr*Ii
    return np.mean((Pc-P)**2 + (Qc-Q)**2), np.abs(Pc-P).max(), np.abs(Qc-Q).max()

mse_d, mp_d, mq_d = kcl_mse(vmag, vang_deg, Y, P_inj, Q_inj)
mse_r, mp_r, mq_r = kcl_mse(vmag, vang_rad, Y, P_inj, Q_inj)
print(f"\nKCL with v_ang as DEGREES (deg2rad applied): MSE={mse_d:.6e}  maxP={mp_d:.4e}  maxQ={mq_d:.4e}")
print(f"KCL with v_ang as RADIANS (no conversion):   MSE={mse_r:.6e}  maxP={mp_r:.4e}  maxQ={mq_r:.4e}")

# Also check: is v_mag correct?
print(f"\nv_mag range: {vmag.min():.4f} - {vmag.max():.4f}")

# Try getting P_spec and Q_spec differently: from generators and loads
gen_p = np.zeros(len(net_buses))
gen_q = np.zeros(len(net_buses))
btoi = {b: i for i, b in enumerate(net_buses)}

for g in net_pf.generators.index:
    bus = net_pf.generators.loc[g, 'bus']
    p = net_pf.generators_t.p.loc[snap, g] if g in net_pf.generators_t.p.columns else 0.
    q = net_pf.generators_t.q.loc[snap, g] if g in net_pf.generators_t.q.columns else 0.
    gen_p[btoi[bus]] += p; gen_q[btoi[bus]] += q

load_p = np.zeros(len(net_buses)); load_q = np.zeros(len(net_buses))
for l in net_pf.loads.index:
    bus = net_pf.loads.loc[l, 'bus']
    p = net_pf.loads_t.p_set.loc[snap, l] if l in net_pf.loads_t.p_set.columns else net_pf.loads.loc[l, 'p_set']
    q = net_pf.loads_t.q_set.loc[snap, l] if l in net_pf.loads_t.q_set.columns else net_pf.loads.loc[l, 'q_set']
    load_p[btoi[bus]] += p; load_q[btoi[bus]] += q

P_net = gen_p - load_p; Q_net = gen_q - load_q
print("\n=== P_net from gen-load vs buses_t.p (first 6 generator buses) ===")
for i, b in enumerate(net_buses[:6]):
    print(f"  {b}: buses_t.p={P_inj[i]:.4f}  gen-load={P_net[i]:.4f}  diff={P_inj[i]-P_net[i]:.4f}")

mse_d2, mp_d2, mq_d2 = kcl_mse(vmag, vang_deg, Y, P_net, Q_net)
print(f"\nKCL with P_net (gen-load), v_ang degrees: MSE={mse_d2:.6e}  maxP={mp_d2:.4e}  maxQ={mq_d2:.4e}")
