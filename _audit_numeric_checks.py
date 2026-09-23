"""Throwaway audit script for tasks 075/076/077 physics-loss audit (2026-09-22).
Read-only numeric verification against the real mixed_1500_w dataset.
Deleted after the audit artifact is written.
"""
import json
import os
import sys
import itertools

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


print("Exec'ing notebook cells 0..15 (through train_power_flow_gnn def)...")
ns, idx = exec_notebook_upto("def train_power_flow_gnn(")
print(f"  -> executed through cell {idx}")

import pypsa  # noqa: E402

PowerFlowDataset = ns["PowerFlowDataset"]
precompute_bfs_order = ns["precompute_bfs_order"]
reconstruct_theta_from_delta = ns["reconstruct_theta_from_delta"]
compute_kvl_loop_closure_loss = ns["compute_kvl_loop_closure_loss"]
compute_balance_dc_local = ns["compute_balance_dc_local"]
compute_balance_dc_global = ns["compute_balance_dc_global"]
compute_flow_loss_dc_local = ns["compute_flow_loss_dc_local"]
compute_physics_residual_edge_local = ns["compute_physics_residual_edge_local"]
PhysicsConfig = ns["PhysicsConfig"]

with open(DATASET_INDEX, "r") as fh:
    meta = json.load(fh)
save_dir = os.path.dirname(DATASET_INDEX)
files = meta["files"]
print(f"Dataset has {len(files)} networks total.")

# Sample indices spanning families: ieee9-like (small idx), cigre14-like (mid), ieee30-like (later)
sample_idx = [0, 1, 2, 300, 301, 302, 600, 601, 602, 900, 901, 902, 1200, 1201, 1202, 1490, 1491, 1492]
sample_idx = sorted(set(i for i in sample_idx if i < len(files)))
print(f"Sampling {len(sample_idx)} networks at indices {sample_idx}")

networks = []
for i in sample_idx:
    net_path = os.path.join(save_dir, files[i])
    networks.append(pypsa.Network(net_path))

# NOTE: PowerFlowDataset builds a flat (net_idx, snapshot_idx) index across ALL
# snapshots of ALL networks -- dataset.get(i) is NOT networks[i] when networks
# have >1 snapshot. Build one single-network dataset per sampled network and
# always take its first snapshot (t_idx=0) to avoid cross-network index bugs.
datasets = [PowerFlowDataset([net], angle_mode="both") for net in networks]

print("\n=== Network family / feature summary ===")
for i, net in zip(sample_idx, networks):
    n_bus = len(net.buses)
    n_trafo = len(net.transformers)
    n_shunt = len(net.shunt_impedances)
    tap_vals = net.transformers["tap_ratio"].values if n_trafo > 0 else np.array([])
    x_trafo = net.transformers["x"].values if n_trafo > 0 else np.array([])
    print(f"  idx={i:5d} buses={n_bus:3d} trafos={n_trafo} shunts={n_shunt} "
          f"tap_range={tap_vals.min() if len(tap_vals) else None}-{tap_vals.max() if len(tap_vals) else None} "
          f"x_trafo_min={x_trafo.min() if len(x_trafo) else None}")

print("\n=== Per-network numeric checks (Tier 0 KVL, Tier 1 DC balance) ===")
results = []
for i, net, ds in zip(sample_idx, networks, datasets):
    data = ds.get(0)  # first snapshot of this single-network dataset
    n_buses = data.x.shape[0]
    y = data.y  # [n_buses,4] = vmag,vang,p,q

    # ── KVL check at TRUE delta_theta ──
    chord_mask_fwd = data.bfs_chord_mask[::2]
    n_chords = int(chord_mask_fwd.sum())
    theta_recon = reconstruct_theta_from_delta(
        data.y_delta_theta, data.bfs_edge_idx, data.bfs_signs, data.bfs_node_order, data.bfs_parent, n_buses
    )
    theta_true_vs_recon_mae = (theta_recon - y[:, 1]).abs().max().item()
    edge_index_fwd = data.edge_index[:, ::2]
    kvl_loss, kvl_diag = compute_kvl_loop_closure_loss(
        data.y_delta_theta, chord_mask_fwd, theta_recon, edge_index_fwd
    )

    # ── Tier 1 DC balance check at TRUE solution ──
    node_pred4 = y.clone()  # [Vmag,Vang,P,Q] = true values everywhere (predicted==truth at PF solution)
    bus_masks = (data.slack_mask, data.pv_mask, data.pq_mask)
    loss_bal_l, mae_bal_l = compute_balance_dc_local(
        data.y_delta_theta, data.edge_index, data.edge_attr, data.x, node_pred4, bus_masks=bus_masks
    )
    loss_bal_g, mae_bal_g = compute_balance_dc_global(
        y[:, 1], data.edge_index, data.edge_attr, data.x, node_pred4, bus_masks=bus_masks
    )

    # Compare to existing accepted DC-local flow-loss error at truth (per-edge, forward lines only)
    dc_mask_fwd = data.dc_flow_mask[::2]
    edge_attr_fwd = data.edge_attr[::2]
    y_line_p = data.y_line_p
    loss_dcl, mae_dcl = compute_flow_loss_dc_local(data.y_delta_theta, edge_attr_fwd, y_line_p, dc_mask_fwd)

    n_trafo = len(net.transformers)
    x_trafo_min = net.transformers["x"].min() if n_trafo > 0 else None
    n_shunt = len(net.shunt_impedances)

    results.append(dict(
        idx=i, n_buses=n_buses, n_chords=n_chords, n_trafo=n_trafo, n_shunt=n_shunt,
        theta_recon_mae=theta_true_vs_recon_mae, kvl_loss=kvl_loss.item(), kvl_mae=kvl_diag["kvl_mae"],
        bal_l_loss=loss_bal_l.item(), bal_l_mae=mae_bal_l.item(),
        bal_g_loss=loss_bal_g.item(), bal_g_mae=mae_bal_g.item(),
        dcl_flow_loss=loss_dcl.item(), dcl_flow_mae=mae_dcl.item(),
        x_trafo_min=x_trafo_min,
    ))

print(f"{'idx':>5} {'buses':>5} {'chords':>6} {'trafo':>5} {'shunt':>5} {'thetaReconMAE':>13} "
      f"{'kvlLoss':>10} {'kvlMAE':>8} {'balL_MAE':>9} {'balG_MAE':>9} {'dcFlowMAE(ref)':>14} {'xTrafoMin':>10}")
for r in results:
    print(f"{r['idx']:5d} {r['n_buses']:5d} {r['n_chords']:6d} {r['n_trafo']:5d} {r['n_shunt']:5d} "
          f"{r['theta_recon_mae']:13.3e} {r['kvl_loss']:10.3e} {r['kvl_mae']:8.3e} "
          f"{r['bal_l_mae']:9.3e} {r['bal_g_mae']:9.3e} {r['dcl_flow_mae']:14.3e} "
          f"{r['x_trafo_min'] if r['x_trafo_min'] is not None else float('nan'):10.4f}")

# ── Demonstrate the "both"-mode P/Q column-misalignment bug in compute_physics_residual_edge_local ──
print("\n=== 'both'-mode compute_physics_residual_edge_local P/Q column misalignment check ===")
# Pick a network with both PV and slack buses (any will do)
i = sample_idx[0]
data = datasets[0].get(0)
n_buses = data.x.shape[0]
y = data.y
delta_theta_pred = data.y_delta_theta  # exact
vmag = y[:, 0]
edge_index = data.edge_index
edge_attr = data.edge_attr
g_diag = data.g_diag
b_diag = data.b_diag
x = data.x
bus_masks = (data.slack_mask, data.pv_mask, data.pq_mask)

# Correct call: node_pred is the 3-col [Vmag,P,Q] edge_delta-style layout
node_pred_3col = torch.stack([y[:, 0], y[:, 2], y[:, 3]], dim=1)  # [Vmag,P,Q]
phys_correct, p_res_correct, q_res_correct = compute_physics_residual_edge_local(
    delta_theta_pred, vmag, node_pred_3col, edge_index, edge_attr, g_diag, b_diag, x,
    bus_masks=bus_masks,
)

# Buggy call (what "both"-mode training loop actually passes): node_pred is the FULL 4-col tensor
node_pred_4col = y.clone()  # [Vmag,Vang,P,Q] -- exactly what "both" mode's node_pred looks like
phys_buggy, p_res_buggy, q_res_buggy = compute_physics_residual_edge_local(
    delta_theta_pred, vmag, node_pred_4col, edge_index, edge_attr, g_diag, b_diag, x,
    bus_masks=bus_masks,
)

slack_idx = int(data.slack_mask.nonzero()[0].item())
pv_idx = data.pv_mask.nonzero().flatten().tolist()
print(f"  network idx={i}, slack bus={slack_idx}, n_pv={len(pv_idx)}")
print(f"  TRUE   P_slack={y[slack_idx,2].item(): .5f}  Q_slack={y[slack_idx,3].item(): .5f}  Vang_slack={y[slack_idx,1].item(): .5f}")
print(f"  correct call: p_res_mean={p_res_correct.item():.4e}  q_res_mean={q_res_correct.item():.4e}  physics_loss={phys_correct.item():.4e}")
print(f"  buggy   call: p_res_mean={p_res_buggy.item():.4e}  q_res_mean={q_res_buggy.item():.4e}  physics_loss={phys_buggy.item():.4e}")
print("  (buggy call reads node_pred[slack,1]=Vang as P and node_pred[slack/pv,2]=P as Q)")
if len(pv_idx) > 0:
    pv0 = pv_idx[0]
    print(f"  PV bus {pv0}: TRUE Q={y[pv0,3].item(): .5f}  TRUE P={y[pv0,2].item(): .5f}  TRUE Vang={y[pv0,1].item(): .5f}")

print("\nDone.")
