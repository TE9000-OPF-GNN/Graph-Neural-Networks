import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Print cell 10 source (PowerFlowDataset)
print("="*80)
print("CELL 10 - PowerFlowDataset")
print("="*80)
src = ''.join(cells[10]['source'])
lines = src.split('\n')
for i, line in enumerate(lines):
    print(f"{i:4d}: {line}")
