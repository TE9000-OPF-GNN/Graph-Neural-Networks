"""List all network component types and counts, check Links, StorageUnits, etc."""
import json, os, copy, logging, numpy as np, pypsa
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))

print("=== Network component counts ===")
components = ['buses', 'lines', 'transformers', 'generators', 'loads',
              'shunt_impedances', 'links', 'storage_units', 'stores', 
              'line_types', 'transformer_types']
for comp in components:
    df = getattr(net0, comp, None)
    if df is not None and len(df) > 0:
        print(f"  {comp}: {len(df)}")

# Check what PyPSA pf() uses: look at what calculate_Y includes
# Does sn.calculate_Y() loop over lines AND transformers AND something else?
import inspect, pypsa.components as pc
src = inspect.getsource(pc.SubNetwork.calculate_Y)
print("\n=== SubNetwork.calculate_Y source snippet ===")
# Print lines mentioning what components are looped over
for line in src.split('\n'):
    if any(x in line for x in ['for', 'lines', 'transformers', 'shunt', 'components', 'iterate']):
        print('  ' + line.rstrip())

# Now check: does calculate_Y handle generators or loads?
print("\n=== Looking for what components Y is built from ===")
for line in src.split('\n'):
    if any(x in line for x in ['iter_components', 'passive', 'branches', 'component']):
        print('  ' + line.rstrip())
