"""Throwaway audit script #2: verify PyG batching correctness for Tier 0 KVL and Tier 1 DC
balance across a real multi-graph batch (different bus counts, transformers/shunts mixed).
"""
import json, os
import numpy as np
import torch

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")
DATA_ROOT = r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN"
DATASET_INDEX = os.path.join(DATA_ROOT, "training_networks_saved", "mixed_1500_w.json")


def load_notebook_cells():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def exec_notebook_upto(marker, ns=None):
    cells = load_notebook_cells()
    ns = ns if ns is not None else {"__name__": "__audit__"}
    idx_target = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and marker in "".join(c.get("source", [])):
            idx_target = i
            break
    assert idx_target is not None, f"marker not found: {marker}"
    for cell in cells[: idx_target + 1]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        exec(compile(src, "<notebook>", "exec"), ns)  # noqa: S102
    return ns, idx_target


ns, idx = exec_notebook_upto("def train_power_flow_gnn(")
import pypsa  # noqa: E402

PowerFlowDataset = ns["PowerFlowDataset"]
collate_with_ptdf = ns["collate_with_ptdf"]
compute_kvl_loop_closure_loss = ns["compute_kvl_loop_closure_loss"]
compute_balance_dc_local = ns["compute_balance_dc_local"]
compute_balance_dc_global = ns["compute_balance_dc_global"]
_reconstruct_batch_theta = ns["_reconstruct_batch_theta"]

with open(DATASET_INDEX, "r") as fh:
    meta = json.load(fh)
save_dir = os.path.dirname(DATASET_INDEX)
files = meta["files"]

# Mix of different sizes/families: ieee9 (idx=1), cigre14-like (idx=302), ieee30-w-shunt (idx=0)
sample_idx = [1, 302, 0]
networks = [pypsa.Network(os.path.join(save_dir, files[i])) for i in sample_idx]

# Individual (unbatched) per-network Data objects
indiv_data = []
for net in networks:
    ds = PowerFlowDataset([net], angle_mode="both")
    indiv_data.append(ds.get(0))

# Batched via the real collate function
batch = collate_with_ptdf(list(indiv_data))
device = torch.device("cpu")

print(f"Batch: {batch.num_graphs} graphs, total nodes={batch.x.shape[0]}, total edges={batch.edge_index.shape[1]}")
node_offsets = [0]
for d in indiv_data:
    node_offsets.append(node_offsets[-1] + d.x.shape[0])
print("node offsets:", node_offsets)

# ── KVL: batched vs per-graph ──
theta_recon_batch = _reconstruct_batch_theta(batch.y_delta_theta_cat if hasattr(batch, "y_delta_theta_cat") else None, batch, device, detach=True) if False else None
# _reconstruct_batch_theta expects delta_theta_pred as a flat tensor across the whole batch
# (concatenated in graph order, same layout as model output) -- build it from the per-graph
# y_delta_theta list already restored by collate_with_ptdf.
delta_theta_pred_batch = torch.cat(batch.y_delta_theta, dim=0)
theta_recon_batch = _reconstruct_batch_theta(delta_theta_pred_batch, batch, device, detach=True)

chord_mask_fwd_batch = batch.bfs_chord_mask[::2]
edge_index_fwd_batch = batch.edge_index[:, ::2]
kvl_loss_batch, kvl_diag_batch = compute_kvl_loop_closure_loss(
    delta_theta_pred_batch, chord_mask_fwd_batch, theta_recon_batch, edge_index_fwd_batch
)
print(f"\nBatched KVL loss={kvl_loss_batch.item():.3e}  mae={kvl_diag_batch['kvl_mae']:.3e}")

# Cross-check: per-graph theta_recon (individually) should match the batched theta_recon,
# offset by each graph's own node range (no cross-graph leakage).
reconstruct_theta_from_delta = ns["reconstruct_theta_from_delta"]
for gi, d in enumerate(indiv_data):
    n_i = d.x.shape[0]
    theta_recon_i = reconstruct_theta_from_delta(
        d.y_delta_theta, d.bfs_edge_idx, d.bfs_signs, d.bfs_node_order, d.bfs_parent, n_i
    )
    theta_recon_from_batch_i = theta_recon_batch[node_offsets[gi]:node_offsets[gi + 1]]
    max_diff = (theta_recon_i - theta_recon_from_batch_i).abs().max().item()
    print(f"  graph {gi} (net_idx={sample_idx[gi]}, n={n_i}): "
          f"max|theta_recon_indiv - theta_recon_from_batch| = {max_diff:.3e}")

# ── Tier 1 DC balance: batched vs per-graph ──
print("\n=== Tier 1 DC balance: batched vs per-graph consistency ===")
node_pred4_batch = batch.y.clone()
bus_masks_batch = (batch.slack_mask, batch.pv_mask, batch.pq_mask)
loss_bal_l_batch, mae_bal_l_batch = compute_balance_dc_local(
    delta_theta_pred_batch, batch.edge_index, batch.edge_attr, batch.x, node_pred4_batch,
    bus_masks=bus_masks_batch,
)
print(f"Batched balance_dc_local: loss={loss_bal_l_batch.item():.4e} mae={mae_bal_l_batch.item():.4e}")

# Per-graph p_calc_dc should match if we slice the batched computation by node range.
# Recompute p_calc_dc manually (replicating the function's internals) to inspect per-node values.
from torch_geometric.utils import scatter
N = batch.x.shape[0]
n_edges_total = batch.edge_index.shape[1]
delta_theta_full = torch.zeros(n_edges_total)
all_fwd = torch.zeros(n_edges_total, dtype=torch.bool)
all_fwd[::2] = True
delta_theta_full[all_fwd] = delta_theta_pred_batch
delta_theta_full[~all_fwd] = -delta_theta_pred_batch
src = batch.edge_index[0]
x_react = batch.edge_attr[:, 1].clamp(min=1e-8)
f_dc = delta_theta_full / x_react
p_calc_dc_batch = scatter(f_dc, src, dim=0, dim_size=N, reduce="sum")

for gi, d in enumerate(indiv_data):
    n_i = d.x.shape[0]
    # per-graph computation using the SAME function on the unbatched Data object
    node_pred4_i = d.y.clone()
    bus_masks_i = (d.slack_mask, d.pv_mask, d.pq_mask)
    _, mae_i = compute_balance_dc_local(d.y_delta_theta, d.edge_index, d.edge_attr, d.x, node_pred4_i, bus_masks=bus_masks_i)
    p_calc_slice = p_calc_dc_batch[node_offsets[gi]:node_offsets[gi + 1]]
    # recompute per-graph p_calc_dc individually for direct numeric comparison
    N_i = d.x.shape[0]
    n_edges_i = d.edge_index.shape[1]
    dtf_i = torch.zeros(n_edges_i)
    fwd_i = torch.zeros(n_edges_i, dtype=torch.bool)
    fwd_i[::2] = True
    dtf_i[fwd_i] = d.y_delta_theta
    dtf_i[~fwd_i] = -d.y_delta_theta
    x_react_i = d.edge_attr[:, 1].clamp(min=1e-8)
    f_dc_i = dtf_i / x_react_i
    p_calc_i = scatter(f_dc_i, d.edge_index[0], dim=0, dim_size=N_i, reduce="sum")
    max_diff = (p_calc_i - p_calc_slice).abs().max().item()
    print(f"  graph {gi} (net_idx={sample_idx[gi]}, n={n_i}): per-graph mae(indiv)={mae_i.item():.4e}  "
          f"max|p_calc_dc(indiv) - p_calc_dc(from batch slice)| = {max_diff:.3e}")

print("\nDone.")
