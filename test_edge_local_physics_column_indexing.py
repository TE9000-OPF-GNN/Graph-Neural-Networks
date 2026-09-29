"""Regression guards for angle_mode="both" column mis-indexing fix in
compute_physics_residual_edge_local (task 079).
See research 2026-09-24-both-mode-edge-local-column-mismatch.md.

Group 0: notebook-source guards (mode-aware column resolution present, edge_delta
unaffected, single call site preserved).
Group 1: synthetic 3-bus fixture, exact by construction - replicated function
copy (must match the notebook function exactly after the fix lands).
"""

import json
import os
import re

import pytest
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
def test_notebook_uses_mode_aware_column_resolution():
    src = _load_cell_source("def compute_physics_residual_edge_local(")
    body = _extract_function_body(src, "compute_physics_residual_edge_local")
    assert "node_pred.shape[1] < 4" in body or "node_pred.shape[1]<4" in body, (
        "compute_physics_residual_edge_local must use mode-aware column resolution "
        "(same pattern as compute_balance_dc_local) - task 079 regression"
    )
    print("TEST 0a ok")


def test_single_call_site_preserved():
    call_count = 0
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        call_count += src.count("_phys_g, _pr_g, _qr_g = compute_physics_residual_edge_local(")
    assert call_count == 1, (
        f"Expected exactly 1 call site (train loop) for compute_physics_residual_edge_local, "
        f"found {call_count} - val/test immunity to this bug class depends on this invariant"
    )
    print("TEST 0b ok")


# -- Replicated helper (must match the notebook function exactly after the fix) -----
def compute_physics_residual_edge_local(
    delta_theta_pred,
    vmag,
    node_pred,
    edge_index,
    edge_attr,
    g_diag,
    b_diag,
    x,
    bus_masks=None,
    use_q_partial_mode=False,
    w_P=0.5,
    w_Q=0.5,
):
    N = vmag.shape[0]
    device = vmag.device

    n_edges_total = edge_index.shape[1]
    delta_theta_full = torch.zeros(n_edges_total, device=device)
    all_fwd = torch.zeros(n_edges_total, dtype=torch.bool, device=device)
    all_fwd[::2] = True

    delta_theta_full[all_fwd] = delta_theta_pred
    delta_theta_full[~all_fwd] = -delta_theta_pred

    src, dst = edge_index[0], edge_index[1]
    v_i = vmag[src]
    v_j = vmag[dst]

    g_s = edge_attr[:, 4]
    b_s = edge_attr[:, 5]

    cos_dt = torch.cos(delta_theta_full)
    sin_dt = torch.sin(delta_theta_full)

    p_edge = -v_i * v_j * (g_s * cos_dt + b_s * sin_dt)
    q_edge = -v_i * v_j * (g_s * sin_dt - b_s * cos_dt)

    p_off = scatter(p_edge, src, dim=0, dim_size=N, reduce="sum")
    q_off = scatter(q_edge, src, dim=0, dim_size=N, reduce="sum")

    p_calc = vmag**2 * g_diag + p_off
    q_calc = -(vmag**2) * b_diag + q_off

    if bus_masks is None:
        slack_mask = x[:, 0] == 1.0
        pv_mask = x[:, 1] == 1.0
        pq_mask = x[:, 2] == 1.0
    else:
        slack_mask, pv_mask, pq_mask = bus_masks

    p_inj = torch.zeros(N, device=device)
    q_inj = torch.zeros(N, device=device)

    p_inj[pq_mask] = x[pq_mask, 3]
    q_inj[pq_mask] = x[pq_mask, 4]
    p_inj[pv_mask] = x[pv_mask, 3]

    # Mode-aware column resolution - mirrors compute_balance_dc_local established
    # pattern (task 076). edge_delta node_pred=[Vmag,P,Q] (3-col); node/both
    # node_pred=[Vmag,Vang,P,Q] (4-col). NEVER hardcode indices without this check.
    _p_col = 1 if node_pred.shape[1] < 4 else 2
    _q_col = _p_col + 1
    q_inj[pv_mask] = node_pred[pv_mask, _q_col]
    p_inj[slack_mask] = node_pred[slack_mask, _p_col]
    q_inj[slack_mask] = node_pred[slack_mask, _q_col]

    b_scale = b_diag.abs().clamp(min=1e-6)
    p_residual = ((p_calc - p_inj) / b_scale) ** 2

    if use_q_partial_mode:
        q_mask = pq_mask | slack_mask
        q_residual = torch.zeros(N, device=device)
        q_residual[q_mask] = ((q_calc[q_mask] - q_inj[q_mask]) / b_scale[q_mask]) ** 2
    else:
        q_residual = ((q_calc - q_inj) / b_scale) ** 2

    p_res_mean = p_residual.mean().detach()
    q_res_mean = q_residual.mean().detach()
    physics_loss = torch.mean(w_P * p_residual + w_Q * q_residual)

    return physics_loss, p_res_mean, q_res_mean


def _make_3bus_fixture(node_pred_cols):
    """3-bus ring: 0=slack, 1=PV, 2=PQ."""

    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    n_edges = edge_index.shape[1]

    r, x_ = 0.01, 0.1
    y = 1.0 / complex(r, x_)

    edge_attr = torch.tensor([[r, x_, 0.0, 1.0, y.real, y.imag]] * n_edges, dtype=torch.float32)
    g_diag = torch.tensor([2 * y.real] * 3)
    b_diag = torch.tensor([2 * y.imag] * 3)

    vmag = torch.tensor([1.00, 1.02, 0.98])
    delta_theta_pred = torch.tensor([0.03, -0.02, 0.01])

    x_feat = torch.zeros(3, 5)
    x_feat[0, 0] = 1.0
    x_feat[1, 1] = 1.0
    x_feat[2, 2] = 1.0

    x_feat[2, 3] = -0.8
    x_feat[2, 4] = -0.3
    x_feat[1, 3] = 0.5

    bus_masks = (x_feat[:, 0].bool(), x_feat[:, 1].bool(), x_feat[:, 2].bool())

    if node_pred_cols == 4:
        node_pred = torch.stack(
            [
                vmag,
                torch.tensor([0.0, 0.07, -0.05]),
                torch.tensor([1.15, 0.5, -0.8]),
                torch.tensor([0.42, 0.55, -0.3]),
            ],
            dim=1,
        )
    else:
        node_pred = torch.stack(
            [
                vmag,
                torch.tensor([1.15, 0.5, -0.8]),
                torch.tensor([0.42, 0.55, -0.3]),
            ],
            dim=1,
        )

    return {
        "delta_theta_pred": delta_theta_pred,
        "vmag": vmag,
        "node_pred": node_pred,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "g_diag": g_diag,
        "b_diag": b_diag,
        "x": x_feat,
        "bus_masks": bus_masks,
    }


# -- Group 1: numeric behavior ------------------------------------------------
def test_both_mode_gradient_routes_to_correct_columns():
    """angle_mode='both' (4-col node_pred): Vang (col 1) must get zero gradient
    from this loss; Q (col 3) at PV/slack must get nonzero gradient."""

    kwargs = _make_3bus_fixture(node_pred_cols=4)
    kwargs["node_pred"] = kwargs["node_pred"].clone().requires_grad_(True)

    physics_loss, _, _ = compute_physics_residual_edge_local(**kwargs)
    physics_loss.backward()
    grad = kwargs["node_pred"].grad

    assert grad[0, 1].item() == 0.0, "Vang column at slack must receive zero gradient"
    assert grad[1, 3].item() != 0.0, "Q column at PV must receive nonzero gradient"
    assert grad[0, 3].item() != 0.0, "Q column at slack must receive nonzero gradient"
    print("TEST 1a ok")


def test_edge_delta_mode_unchanged():
    """3-col node_pred (edge_delta): _p_col/_q_col must reduce to (1,2) - bit-identical
    to the pre-fix hardcoded behavior. Golden values captured from the pre-fix function
    on this exact fixture."""

    kwargs = _make_3bus_fixture(node_pred_cols=3)
    _, p_res, q_res = compute_physics_residual_edge_local(**kwargs)

    assert p_res.item() == pytest.approx(0.002421528100967407, abs=1e-9)
    assert q_res.item() == pytest.approx(0.000253315898589790, abs=1e-9)
    print("TEST 1b ok")
