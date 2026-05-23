import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Cell 14 - training loop and physics loss
print("="*80)
print("CELL 14 (first 300 lines)")
print("="*80)
src = ''.join(cells[14]['source'])
lines = src.split('\n')
for i, line in enumerate(lines[:300]):
    print(f"{i:4d}: {line}")
print(f"\n... total {len(lines)} lines in cell 14")
