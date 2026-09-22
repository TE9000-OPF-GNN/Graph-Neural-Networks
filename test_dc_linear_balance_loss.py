"""Regression guards for Tier 1 DC-linear power balance (KCL) loss (task 076).
See research 2026-09-21-node-power-balance-loss-from-predicted-flows.md, Finding 5b.

Group 0: notebook-source guards (functions defined, shunt/tap-free, wired into loops).
Group 1: synthetic 3-bus radial network, exact by construction (loss ~0 at true solution).
"""
import json
import os
import re

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")


def _load_cell_source(marker):
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"cell containing {marker!r} not found")


def _extract_function_body(src, func_name):
    start = src.index(f"def {func_name}(")
    rest = src[start:]
    m = re.search(r"\ndef \w+\(", rest[1:])
    return rest[: m.start() + 1] if m else rest


# -- Group 0: notebook-source guards ------------------------------------------
def test_notebook_defines_balance_functions():
    src = _load_cell_source("def compute_flow_loss_dc_global(")
    assert "def compute_balance_dc_local(" in src
    assert "def compute_balance_dc_global(" in src
    print("TEST 0a ok")


def test_balance_functions_never_touch_shunt_or_tap():
    src = _load_cell_source("def compute_balance_dc_local(")
    for fn in ("compute_balance_dc_local", "compute_balance_dc_global"):
        body = _extract_function_body(src, fn)
        for banned in ("g_diag", "b_diag", "edge_attr[:, 3]", "edge_attr[:, 4]", "edge_attr[:, 5]"):
            assert banned not in body, f"{fn} references {banned!r} (breaks shunt/tap immunity)"
        assert "edge_attr[:, 1]" in body, f"{fn} must use reactance column 1"
    print("TEST 0b ok")


def test_notebook_wires_balance_in_train_and_val_loops():
    src = _load_cell_source("def train_power_flow_gnn(")
    assert "compute_balance_dc_local(" in src
    assert "compute_balance_dc_global(" in src
    assert "physics_cfg.enable_balance_dc" in src
    print("TEST 0c ok")


def test_no_hardcoded_first_two_edge_slice():
    src = _load_cell_source("def compute_balance_dc_local(")
    body = (_extract_function_body(src, "compute_balance_dc_local")
            + _extract_function_body(src, "compute_balance_dc_global"))
    for bad in ("edge_index[:, :2]", "edge_attr[:2]"):
        assert bad not in body, f"regression marker {bad!r} found (MANDATORY RULE 7)"
    print("TEST 0d ok")


def test_notebook_config_has_enable_balance_dc_field():
    src = _load_cell_source("class PhysicsConfig:")
    assert "enable_balance_dc" in src
    print("TEST 0e ok")


# -- Replicated helpers (must match the notebook functions exactly) -----------
def compute_balance_dc_local(delta_theta_pred, edge_index, edge_attr, x, node_pred, bus_masks=None):
    N = x.shape[0]
    device = delta_theta_pred.device
    n_edges_total = edge_index.shape[1]
    delta_theta_full = torch.zeros(n_edges_total, device=device)
    all_fwd = torch.zeros(n_edges_total, dtype=torch.bool, device=device)
    all_fwd[::2] = True
    delta_theta_full[all_fwd] = delta_theta_pred
    delta_theta_full[~all_fwd] = -delta_theta_pred
    src = edge_index[0]
    x_react = edge_attr[:, 1].clamp(min=1e-8)
    f_dc = delta_theta_full / x_react
    p_calc_dc = scatter(f_dc, src, dim=0, dim_size=N, reduce="sum")
    if bus_masks is None:
        slack_mask, pv_mask, pq_mask = x[:, 0] == 1.0, x[:, 1] == 1.0, x[:, 2] == 1.0
    else:
        slack_mask, pv_mask, pq_mask = bus_masks
    p_inj = torch.zeros(N, device=device)
    p_inj[pq_mask] = x[pq_mask, 3]
    p_inj[pv_mask] = x[pv_mask, 3]
    p_col = 1 if node_pred.shape[1] < 4 else 2
    p_inj[slack_mask] = node_pred[slack_mask, p_col]
    return F.mse_loss(p_calc_dc, p_inj), (p_calc_dc - p_inj).abs().mean().detach()


def compute_balance_dc_global(theta_pred, edge_index, edge_attr, x, node_pred, bus_masks=None):
    N = x.shape[0]
    device = theta_pred.device
    src, dst = edge_index[0], edge_index[1]
    x_react = edge_attr[:, 1].clamp(min=1e-8)
    f_dc = (theta_pred[src] - theta_pred[dst]) / x_react
    p_calc_dc = scatter(f_dc, src, dim=0, dim_size=N, reduce="sum")
    if bus_masks is None:
        slack_mask, pv_mask, pq_mask = x[:, 0] == 1.0, x[:, 1] == 1.0, x[:, 2] == 1.0
    else:
        slack_mask, pv_mask, pq_mask = bus_masks
    p_inj = torch.zeros(N, device=device)
    p_inj[pq_mask] = x[pq_mask, 3]
    p_inj[pv_mask] = x[pv_mask, 3]
    p_col = 1 if node_pred.shape[1] < 4 else 2
    p_inj[slack_mask] = node_pred[slack_mask, p_col]
    return F.mse_loss(p_calc_dc, p_inj), (p_calc_dc - p_inj).abs().mean().detach()


# -- Group 1: synthetic 3-bus radial network, exact by construction -----------
def _synthetic_dc_consistent_case():
    theta = torch.tensor([0.0, -0.05, -0.12])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])   # fwd,rev,fwd,rev
    edge_attr = torch.zeros(4, 6)
    edge_attr[:, 1] = torch.tensor([0.1, 0.1, 0.2, 0.2])
    delta_theta_pred = torch.tensor([theta[0] - theta[1], theta[1] - theta[2]])  # fwd only

    x_feat = torch.zeros(3, 6)
    x_feat[0, 0] = 1.0   # slack
    x_feat[1, 2] = 1.0   # PQ
    x_feat[2, 2] = 1.0   # PQ

    f01 = (theta[0] - theta[1]) / 0.1
    f12 = (theta[1] - theta[2]) / 0.2
    p_true = torch.tensor([f01, -f01 + f12, -f12])   # balance exact by construction
    x_feat[1, 3] = p_true[1]
    x_feat[2, 3] = p_true[2]

    node_pred = torch.zeros(3, 4)
    node_pred[:, 1] = theta
    node_pred[0, 2] = p_true[0]
    bus_masks = (x_feat[:, 0] == 1.0, x_feat[:, 1] == 1.0, x_feat[:, 2] == 1.0)
    return delta_theta_pred, edge_index, edge_attr, x_feat, node_pred, bus_masks, theta


def test_balance_dc_local_zero_at_true_solution():
    delta_theta_pred, edge_index, edge_attr, x_feat, node_pred, bus_masks, _ = _synthetic_dc_consistent_case()
    loss, mae = compute_balance_dc_local(delta_theta_pred, edge_index, edge_attr, x_feat, node_pred, bus_masks)
    assert loss.item() < 1e-8 and mae.item() < 1e-4
    print("TEST 1a ok")


def test_balance_dc_global_zero_at_true_solution():
    _, edge_index, edge_attr, x_feat, node_pred, bus_masks, theta = _synthetic_dc_consistent_case()
    loss, mae = compute_balance_dc_global(theta, edge_index, edge_attr, x_feat, node_pred, bus_masks)
    assert loss.item() < 1e-8 and mae.item() < 1e-4
    print("TEST 1b ok")


if __name__ == "__main__":
    test_notebook_defines_balance_functions()
    test_balance_functions_never_touch_shunt_or_tap()
    test_notebook_wires_balance_in_train_and_val_loops()
    test_no_hardcoded_first_two_edge_slice()
    test_notebook_config_has_enable_balance_dc_field()
    test_balance_dc_local_zero_at_true_solution()
    test_balance_dc_global_zero_at_true_solution()
    print("\nAll DC-linear balance (Tier 1) tests passed.")
