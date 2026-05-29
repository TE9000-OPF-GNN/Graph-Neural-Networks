import json

f = open(r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb', 'r', encoding='utf-8')
nb = json.load(f)
f.close()

# Cell 22: evaluate_models_on_networks + plot_model_metrics_across_networks
cell22 = nb['cells'][22]
src = cell22['source']
print("=== CELL 22 (lines 0-440) ===")
for i, line in enumerate(src):
    print(f"L{i}: {line}", end='')

print("\n\n=== CELL 23 (_collect_test_set_predictions) ===")
cell23 = nb['cells'][23]
src23 = cell23['source']
for i, line in enumerate(src23):
    print(f"L{i}: {line}", end='')
