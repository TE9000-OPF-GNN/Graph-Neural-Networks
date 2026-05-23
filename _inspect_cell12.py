import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Cell 12 - should have physics_informed_loss_batch
print("="*80)
print("CELL 12 - first 200 lines")
print("="*80)
src = ''.join(cells[12]['source'])
lines = src.split('\n')
for i, line in enumerate(lines[:200]):
    print(f"{i:4d}: {line}")
print(f"\n... total {len(lines)} lines in cell 12")
