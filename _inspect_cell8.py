import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Print cell 8 source (compute_ptdf_matrix lives here)
print("="*80)
print("CELL 8 - Power System Helpers")
print("="*80)
src8 = ''.join(cells[8]['source'])
# Print first 200 lines
lines8 = src8.split('\n')
for i, line in enumerate(lines8[:250]):
    print(f"{i:4d}: {line}")
print(f"\n... (total {len(lines8)} lines in cell 8)")
