import json, os, copy, logging
import numpy as np
import pypsa
logging.disable(logging.CRITICAL)

DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
files = d['files']

full = os.path.join(NETS_DIR, files[0])
net = pypsa.Network(full)
buses = list(net.buses.index)
print("Buses:", buses)
print("Num lines:", len(net.lines), "  Num transformers:", len(net.transformers))
print("Shunts:", net.shunt_impedances[['bus','b','g']].to_string())
print()

# Look for lines with extreme admittance
print("=== Lines with |y_series| > 100 ===")
for line in net.lines.index:
    row = net.lines.loc[line]
    r, x = row['r'], row['x']
    z = complex(r, x)
    if abs(z) < 1e-12:
        print(f"  {line}: ZERO impedance bus0={row['bus0']} bus1={row['bus1']}")
        continue
    y_mag = abs(1.0/z)
    if y_mag > 100:
        print(f"  {line}: |y|={y_mag:.1f}  r={r:.6f}  x={x:.6f}  bus0={row['bus0']}  bus1={row['bus1']}")

print()
print("=== Transformers with |y_series| > 100 ===")
for t in net.transformers.index:
    row = net.transformers.loc[t]
    if 'x_pu_eff' in row.index and not (row['x_pu_eff'] != row['x_pu_eff']) and float(row['x_pu_eff']) != 0:
        x = float(row['x_pu_eff'])
        r = float(row['r_pu_eff']) if 'r_pu_eff' in row.index else 0.0
    else:
        x = float(row['x'])
        r = float(row.get('r', 0.0))
    z = complex(r, x)
    if abs(z) < 1e-12:
        print(f"  {t}: ZERO impedance")
        continue
    y_mag = abs(1.0/z)
    if y_mag > 100:
        print(f"  {t}: |y|={y_mag:.1f}  r={r:.6f}  x={x:.6f}  tap={row.get('tap_ratio',1.0)}  bus0={row['bus0']}  bus1={row['bus1']}")

print()
print("=== Lines connected to VL6_0 ===")
vl6 = 'VL6_0'
lines_vl6 = net.lines[(net.lines.bus0 == vl6) | (net.lines.bus1 == vl6)]
print(lines_vl6[['bus0','bus1','r','x','b']].to_string())
print()
trafos_vl6 = net.transformers[(net.transformers.bus0 == vl6) | (net.transformers.bus1 == vl6)]
cols = [c for c in ['bus0','bus1','x','r','tap_ratio','x_pu_eff','r_pu_eff'] if c in net.transformers.columns]
print("Transformers at VL6_0:")
print(trafos_vl6[cols].to_string())
