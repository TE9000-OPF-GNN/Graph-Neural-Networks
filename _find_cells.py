import json

f = open(r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb', 'r', encoding='utf-8')
nb = json.load(f)
f.close()

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    nlines = len(cell.get('source', []))
    if 'def plot_model_metrics_across_networks' in src:
        print(f"Cell {i}: plot_model_metrics_across_networks, lines={nlines}")
    if 'def _collect_test_set_predictions' in src:
        print(f"Cell {i}: _collect_test_set_predictions, lines={nlines}")
    if 'def check_hparam_results' in src:
        print(f"Cell {i}: check_hparam_results, lines={nlines}")
    if 'def evaluate_models_on_networks' in src:
        print(f"Cell {i}: evaluate_models_on_networks, lines={nlines}")
