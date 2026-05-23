import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[14]['source'])
lines = src.split('\n')

# Find physics_informed_loss_batch
for i, line in enumerate(lines):
    if 'def physics_informed_loss_batch' in line:
        print(f"physics_informed_loss_batch starts at line {i}")
        for j in range(i, min(i+120, len(lines))):
            print(f"{j:4d}: {lines[j]}")
        break
