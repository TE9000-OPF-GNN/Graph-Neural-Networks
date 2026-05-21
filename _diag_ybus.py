import json, os, copy, logging
import numpy as np
import pypsa
logging.disable(logging.CRITICAL)

DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
files = d['files']

def compute_Y_manual(network):
    buses = list(network.buses.index)
    bus_to_idx = {b: i for i, b in enumerate(buses)}
    Y = np.zeros((len(buses), len(buses)), dtype=complex)
    for line in network.lines.index:
        row = network.lines.loc[line]
        r, x = row['r'], row['x']
        b = row['b'] if 'b' in network.lines.columns else 0.0
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y_se = 1.0 / z
        y_sh = complex(0., b / 2.)
        fi, ti = bus_to_idx[row['bus0']], bus_to_idx[row['bus1']]
        Y[fi, fi] += y_se + y_sh
        Y[ti, ti] += y_se + y_sh
        Y[fi, ti] -= y_se
        Y[ti, fi] -= y_se
    for t in network.transformers.index:
        row = network.transformers.loc[t]
        has_pu_eff = ('x_pu_eff' in row.index and
                      not (row['x_pu_eff'] != row['x_pu_eff']) and
                      float(row['x_pu_eff']) != 0.0)
        if has_pu_eff:
            x = float(row['x_pu_eff'])
            r = float(row['r_pu_eff']) if 'r_pu_eff' in row.index else 0.0
        else:
            x = float(row['x'])
            r = float(row.get('r', 0.0))
        tap = float(row['tap_ratio']) if 'tap_ratio' in row.index else 1.0
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y_se = 1.0 / z
        fi, ti = bus_to_idx[row['bus0']], bus_to_idx[row['bus1']]
        tc = complex(tap, 0.)
        Y[fi, fi] += y_se / (tc * tc.conjugate())
        Y[ti, ti] += y_se
        Y[fi, ti] -= y_se / tc.conjugate()
        Y[ti, fi] -= y_se / tc
    return Y

def compute_Y_pypsa(net):
    net2 = copy.deepcopy(net)
    net2.determine_network_topology()
    sn_label = list(net2.sub_networks.index)[0]
    sn = net2.sub_networks.at[sn_label, 'obj']
    sn.calculate_Y()
    return np.asarray(sn.Y.todense())

# ---- Batch comparison across all 100 sampled networks ----
max_diffs = []
max_diffs_shunt = []
max_diffs_noshunt = []

for fpath in files[:100]:
    full = os.path.join(NETS_DIR, fpath)
    net = pypsa.Network(full)
    Ym = compute_Y_manual(net)
    Yp = compute_Y_pypsa(net)
    md = float(np.abs(Ym - Yp).max())
    max_diffs.append(md)
    if len(net.shunt_impedances) > 0:
        max_diffs_shunt.append(md)
    else:
        max_diffs_noshunt.append(md)

print("=== Y-Matrix Discrepancy Summary (100 networks) ===")
print(f"  All:       max={max(max_diffs):.4e}  mean={sum(max_diffs)/len(max_diffs):.4e}")
print(f"  Shunts:    n={len(max_diffs_shunt):3d}  max={max(max_diffs_shunt) if max_diffs_shunt else 0:.4e}  mean={(sum(max_diffs_shunt)/len(max_diffs_shunt)) if max_diffs_shunt else 0:.4e}")
print(f"  No shunts: n={len(max_diffs_noshunt):3d}  max={max(max_diffs_noshunt):.4e}  mean={sum(max_diffs_noshunt)/len(max_diffs_noshunt):.4e}")

# ---- Detail on first shunt network ----
for fpath in files[:100]:
    full = os.path.join(NETS_DIR, fpath)
    net = pypsa.Network(full)
    if len(net.shunt_impedances) == 0:
        continue
    buses = list(net.buses.index)
    Ym = compute_Y_manual(net)
    Yp = compute_Y_pypsa(net)
    diff = np.abs(Ym - Yp)
    i_max, j_max = np.unravel_index(diff.argmax(), diff.shape)
    diag_pypsa = np.abs(np.diag(Yp))

    print("\n=== First shunt network detail ===")
    print(f"  File: {fpath}")
    print(f"  Buses: {len(buses)}, Shunts: {len(net.shunt_impedances)}")
    print(f"  Shunt data:\n{net.shunt_impedances[['bus','b','g']].to_string()}")
    print(f"  Worst diff at [{buses[i_max]}, {buses[j_max]}]  diff={diff[i_max,j_max]:.4e}")
    print(f"    manual  = {Ym[i_max,j_max]:.6f}")
    print(f"    pypsa   = {Yp[i_max,j_max]:.6f}")
    print(f"    delta   = {Yp[i_max,j_max] - Ym[i_max,j_max]:.6f}")
    print(f"  Diagonal |Y| at that bus = {diag_pypsa[i_max]:.4f}")
    print(f"  Relative diagonal error  = {100 * diff[i_max, i_max] / max(diag_pypsa[i_max], 1e-12):.1f}%")

    # Show shunt contribution explicitly
    for si in net.shunt_impedances.index:
        sr = net.shunt_impedances.loc[si]
        b_val = float(sr['b']) if 'b' in sr.index else 0.0
        g_val = float(sr['g']) if 'g' in sr.index else 0.0
        y_sh = complex(g_val, b_val)
        print(f"  Shunt '{si}' at bus '{sr['bus']}': y = {y_sh:.6f}  (missing from manual Y diagonal)")
    break
