import json

with open(r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print()
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type'][:2].upper()
    src = ''.join(cell['source'])
    first_line = src.strip().split('\n')[0][:100] if src.strip() else '(empty)'
    print(f"Cell {i:3d} [{ct}]: {first_line}")
