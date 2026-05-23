import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    ct = c["cell_type"]
    first_line = c["source"][0].strip() if c["source"] else "(empty)"
    print(f"Cell {i:2d} ({ct:8s}): {first_line[:90]}")
