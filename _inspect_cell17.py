import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[17]['source'])
lines = src.split('\n')
print(f"Cell 17: total {len(lines)} lines")
for i, line in enumerate(lines[:150]):
    print(f"{i:4d}: {line}")
