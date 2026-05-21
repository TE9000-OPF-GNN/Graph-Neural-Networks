"""Check if P_calc from branch flows matches buses_t.p vs Y*V"""
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
btoi = {b: i for i, b in enumerate(net_buses)}
vmag = net_pf.buses_t.v_mag_pu.loc[snap].reindex(net_buses).values
vang = np.deg2rad(net_pf.buses_t.v_ang.loc[snap].reindex(net_buses).values)
V = vmag * np.exp(1j * vang)
P_inj_buses = net_pf.buses_t.p.loc[snap].reindex(net_buses).values
Q_inj_buses = net_pf.buses_t.q.loc[snap].reindex(net_buses).values

# Compute S from individual branch flows (truth check)
P_branch = np.zeros(len(net_buses))
Q_branch = np.zeros(len(net_buses))

# Lines
for l in net_pf.lines.index:
    row = net_pf.lines.loc[l]
    fi, ti = btoi[row.bus0], btoi[row.bus1]
    Vi, Vj = V[fi], V[ti]
    r, x, b = float(row.r), float(row.x), float(row.get('b', 0.))
    y_se = 1. / complex(r, x) if abs(complex(r, x)) > 1e-15 else 0.
    y_sh = complex(0., b / 2.)
    # Power flowing from bus i: S_i = V_i * conj(y_se*(V_i-V_j) + y_sh*V_i)
    I_from = y_se * (Vi - Vj) + y_sh * Vi
    S_from = Vi * np.conj(I_from)
    I_to = y_se * (Vj - Vi) + y_sh * Vj
    S_to = Vj * np.conj(I_to)
    P_branch[fi] -= S_from.real  # power injected INTO bus fi from network
    Q_branch[fi] -= S_from.imag
    P_branch[ti] -= S_to.real
    Q_branch[ti] -= S_to.imag

# Transformers (pi model, tap on bus0)
for t in net_pf.transformers.index:
    row = net_pf.transformers.loc[t]
    fi, ti = btoi[row.bus0], btoi[row.bus1]
    Vi, Vj = V[fi], V[ti]
    x = float(row.get('x_pu_eff', row.x))
    r = float(row.get('r_pu_eff', row.r) if 'r_pu_eff' in row.index else row.get('r', 0.))
    tap = float(row.get('tap_ratio', 1.0))
    z = complex(r, x)
    if abs(z) < 1e-15: continue
    y_se = 1. / z
    # Standard pi model: bus0 has tap
    I_from = y_se / tap * (Vi / tap - Vj)  # current INTO the network from fi
    I_to = y_se * (Vj - Vi / tap)
    S_from = Vi * np.conj(I_from)
    S_to = Vj * np.conj(I_to)
    P_branch[fi] -= S_from.real
    Q_branch[fi] -= S_from.imag
    P_branch[ti] -= S_to.real
    Q_branch[ti] -= S_to.imag

# Shunts
for si in net_pf.shunt_impedances.index:
    row = net_pf.shunt_impedances.loc[si]
    bi = btoi[row.bus]
    b_val = float(row.get('b', 0.))
    g_val = float(row.get('g', 0.))
    Vi = V[bi]
    y_sh = complex(g_val, b_val)
    S_sh = Vi * np.conj(y_sh * Vi)
    P_branch[bi] -= S_sh.real
    Q_branch[bi] -= S_sh.imag

print("=== buses_t.p vs P_from_branches vs KCL ===")
print(f"{'Bus':<12} {'buses_t.p':>10} {'branch_sum':>10} {'diff':>8}")
for i, bus in enumerate(net_buses):
    diff = abs(P_branch[i] - P_inj_buses[i])
    if diff > 0.001:
        print(f"{bus:<12} {P_inj_buses[i]:10.4f} {P_branch[i]:10.4f} {diff:8.4f}")

total_diff = np.max(np.abs(P_branch - P_inj_buses))
print(f"\nMax |P_branch - buses_t.p|: {total_diff:.6e}")
print(f"Max |Q_branch - buses_t.q|: {np.max(np.abs(Q_branch - Q_inj_buses)):.6e}")

# Now check generators/loads directly
print("\n=== Generators at VL1_0 and VL2_0 ===")
for g in net_pf.generators.index:
    bus = net_pf.generators.loc[g, 'bus']
    if bus in ['VL1_0', 'VL2_0']:
        p = net_pf.generators_t.p.loc[snap, g] if g in net_pf.generators_t.p.columns else 'N/A'
        print(f"  Gen {g} at {bus}: p={p}")

print("\n=== Loads at VL1_0 and VL2_0 ===")
for l in net_pf.loads.index:
    bus = net_pf.loads.loc[l, 'bus']
    if bus in ['VL1_0', 'VL2_0']:
        p_set = net_pf.loads_t.p_set.loc[snap, l] if l in net_pf.loads_t.p_set.columns else net_pf.loads.loc[l, 'p_set']
        print(f"  Load {l} at {bus}: p_set={p_set}")
