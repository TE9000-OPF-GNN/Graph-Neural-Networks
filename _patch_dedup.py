"""Patch: eliminate redundant evaluate_models_on_networks call.
- plot_model_metrics_across_networks accepts optional pre-computed metrics
- check_hparam_results passes multi_net_metrics to avoid recompute
"""
import json

NB_PATH = r"c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─── Patch A: plot_model_metrics_across_networks signature + body (cell 22) ───
cell22 = nb["cells"][22]
src = cell22["source"]

# Find the def line
for k, line in enumerate(src):
    if 'def plot_model_metrics_across_networks(' in line:
        # Replace signature to add metrics=None param
        old_sig = src[k]
        new_sig = old_sig.replace(
            'def plot_model_metrics_across_networks(models_dict, networks, title_prefix="Multi-net", use_pnom_share=None):',
            'def plot_model_metrics_across_networks(models_dict, networks, title_prefix="Multi-net", use_pnom_share=None, metrics=None):'
        )
        src[k] = new_sig
        
        # Next line is the evaluate call - make it conditional
        assert 'metrics = evaluate_models_on_networks' in src[k+1], f"Unexpected line at {k+1}: {src[k+1]}"
        src[k+1] = '    if metrics is None:\n        metrics = evaluate_models_on_networks(models_dict, networks, use_pnom_share=use_pnom_share)\n'
        break

cell22["source"] = src
print("Patch A: plot_model_metrics_across_networks accepts pre-computed metrics")

# ─── Patch B: check_hparam_results passes metrics (cell 17) ───
cell17 = nb["cells"][17]
src17 = cell17["source"]

# Find the plot_model_metrics_across_networks call
for k, line in enumerate(src17):
    if 'plot_model_metrics_across_networks(' in line:
        # Find the closing paren of this call (should be a few lines down)
        # Current call:
        #   plot_model_metrics_across_networks(
        #       models_dict,
        #       eval_networks,
        #       title_prefix=title_prefix,
        #       use_pnom_share=use_pnom_share,
        #   )
        # Need to add metrics=multi_net_metrics before the closing )
        # Find the line with "use_pnom_share=use_pnom_share,"
        for j in range(k+1, k+10):
            if 'use_pnom_share=use_pnom_share,' in src17[j]:
                # Add metrics param after this line
                indent = src17[j][:len(src17[j]) - len(src17[j].lstrip())]
                src17.insert(j+1, f'{indent}metrics=multi_net_metrics,\n')
                break
        break

cell17["source"] = src17
print("Patch B: check_hparam_results passes pre-computed metrics")

# ─── Write back ───
with open(NB_PATH, "w", encoding="utf-8", newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nDone. Redundant evaluation eliminated.")
