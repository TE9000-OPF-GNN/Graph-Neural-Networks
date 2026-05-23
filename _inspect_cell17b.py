import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[17]['source'])
lines = src.split('\n')
for i in range(150, len(lines)):
    print(f"{i:4d}: {lines[i]}")
