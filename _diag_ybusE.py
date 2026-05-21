"""Inspect calculate_dependent_values and check lines x_pu vs x"""
import json, os, copy, logging, numpy as np, pypsa, inspect
import pypsa.pf as ppf
logging.disable(logging.CRITICAL)
DATA_ROOT = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN'
NETS_DIR = os.path.join(DATA_ROOT, 'training_networks_saved')
with open(os.path.join(NETS_DIR, 'mixed_1500_wide.json')) as f:
    d = json.load(f)
net0 = pypsa.Network(os.path.join(NETS_DIR, d['files'][0]))

# Print calculate_dependent_values source
src = inspect.getsource(ppf.calculate_dependent_values)
print("=== calculate_dependent_values source ===")
print(src)
