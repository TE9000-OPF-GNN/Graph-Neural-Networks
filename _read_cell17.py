import json

f = open(r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb', 'r', encoding='utf-8')
nb = json.load(f)
f.close()

cell17 = nb['cells'][17]
src = cell17['source']
# Print lines 815-882
for i in range(815, min(882, len(src))):
    print(f"L{i}: {src[i]}", end='')
