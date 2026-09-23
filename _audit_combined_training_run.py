"""Throwaway audit script #3: real short training run with ALL THREE new loss terms
active simultaneously (angle_mode="both", fraction_kvl>0, enable_balance_dc=True,
physics/dcf_local/dcf_global all active from epoch 0) -- the exact combined-activation
scenario the audit was asked to check, which has never been run before.
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


# Need evaluate_gnn_on_test_set too, since train_power_flow_gnn calls it internally.
ns, idx = exec_notebook_upto("def evaluate_gnn_on_test_set(")
import pypsa  # noqa: E402

PhysicsConfig = ns["PhysicsConfig"]
train_power_flow_gnn = ns["train_power_flow_gnn"]

with open(DATASET_INDEX, "r") as fh:
    meta = json.load(fh)
save_dir = os.path.dirname(DATASET_INDEX)
files = meta["files"]

sample_idx = [0, 1, 2, 300, 301, 302, 600, 601, 602, 900, 901, 902, 1200, 1201]
networks = [pypsa.Network(os.path.join(save_dir, files[i])) for i in sample_idx]

cfg = PhysicsConfig(
    loss_weight_mode="adaptive",
    fraction_physics=0.1,
    fraction_dcf_local=0.1,
    fraction_dcf_global=0.1,
    fraction_kvl=0.1,
    enable_balance_dc=True,
    use_ptdf_loss=False,
)

print("Running 3 epochs, angle_mode='both', ALL of KVL+Tier1-balance+Tier2-physics+dcf_local/global active from epoch 0...")
model, history, test_metrics = train_power_flow_gnn(
    networks=networks,
    num_epochs=3,
    angle_mode="both",
    physics_cfg=cfg,
    physics_activation_start=0,
    dcf_local_activation_start=0,
    dcf_global_activation_start=0,
    kvl_activation_start=0,
    activation_ramp_epochs=2,
    batch_size=2,
    seed=123,
)

print("\n=== History finiteness / trend check ===")
for key in ("train_loss", "train_mse", "train_physics", "train_kvl_loss", "val_kvl_loss",
            "train_flow_loss", "val_flow_loss", "train_eff_w_kvl", "train_eff_w_flow",
            "train_eff_w_phys", "train_delta_theta"):
    if key not in history:
        print(f"  {key}: NOT IN HISTORY")
        continue
    vals = history[key]
    finite = all(np.isfinite(v) for v in vals)
    print(f"  {key}: {vals}  finite={finite}")

print("\nDone.")
