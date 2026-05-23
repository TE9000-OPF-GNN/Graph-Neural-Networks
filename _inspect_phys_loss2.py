import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[11]['source'])
lines = src.split('\n')

# Print from line 385 onwards
for i in range(385, min(520, len(lines))):
    print(f"{i:4d}: {lines[i]}")
print(f"\n... total {len(lines)} lines in cell 11")
