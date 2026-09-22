"""Regression guards for task 077: angle_mode="both" nodal Y-bus balance addition.

Group 0: notebook-source guards (new call present + additive, edge-local call untouched,
          no delta_theta_pred reference, gated strictly on angle_mode=="both", appears once).
Group 1: functional gradient test using the real (notebook-extracted) physics_informed_loss_batch
          on a tiny synthetic 3-bus network -- proves gradients reach all 4 node_pred columns.
"""
import json
import os

import torch

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")


def _load_notebook_cells():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _load_train_cell_source():
    for cell in _load_notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def train_power_flow_gnn" in src:
            return src
    raise AssertionError("train_power_flow_gnn cell not found")


# ── Group 0: notebook-source guards ──────────────────────────────────────────
def test_edge_local_block_unchanged():
    src = _load_train_cell_source()
    assert "if _edge_local_physics is not None:" in src
    assert "physics = _edge_local_physics" in src
    assert "p_res = _edge_local_p_res" in src
    assert "q_res = _edge_local_q_res" in src
    print("TEST 0a ok")


def test_new_nodal_call_present_and_additive():
    src = _load_train_cell_source()
    marker = 'if angle_mode == "both" and (physics_cfg.w_phys > 0.0 or physics_cfg.loss_weight_mode == "adaptive"):'
    assert src.count(marker) == 1, f"expected exactly 1 occurrence, found {src.count(marker)}"
    assert "physics = physics + _nodal_physics" in src
    assert "p_res   = p_res + _nodal_p_res" in src or "p_res = p_res + _nodal_p_res" in src
    assert "q_res   = q_res + _nodal_q_res" in src or "q_res = q_res + _nodal_q_res" in src
    print("TEST 0b ok")


def test_new_call_never_references_delta_theta_pred():
    src = _load_train_cell_source()
    start_marker = "[STSI 220926]: task 077"
    end_marker = "q_res   = q_res + _nodal_q_res"
    start = src.index(start_marker)
    end = src.index(end_marker) + len(end_marker)
    window = src[start:end]
    assert "delta_theta_pred" not in window, \
        "nodal balance call must not reference delta_theta_pred"
    print("TEST 0c ok")


def test_gated_strictly_on_both_mode_train_loop_only():
    src = _load_train_cell_source()
    # exactly one occurrence -> only in the training per-batch loop, not val/test
    marker = 'if angle_mode == "both" and (physics_cfg.w_phys > 0.0 or physics_cfg.loss_weight_mode == "adaptive"):'
    assert src.count(marker) == 1
    # positioned after the edge-local dispatch block, before node_batch assembly
    idx_block = src.index("if _edge_local_physics is not None:")
    idx_new = src.index(marker)
    idx_node_batch = src.index("node_batch = batch.batch", idx_new)
    assert idx_block < idx_new < idx_node_batch
    print("TEST 0d ok")


# ── Group 1: functional gradient test (real notebook functions) ─────────────
def _exec_notebook_defs():
    """Exec notebook cells up to and including physics_informed_loss_batch's definition."""
    cells = _load_notebook_cells()
    idx_target = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and "def physics_informed_loss_batch(" in "".join(c.get("source", [])):
            idx_target = i
            break
    assert idx_target is not None, "physics_informed_loss_batch cell not found"

    ns = {"__name__": "__task077_test__"}
    for cell in cells[:idx_target + 1]:
        if cell.get("cell_type") != "code":
            continue
        exec(compile("".join(cell.get("source", [])), "<notebook>", "exec"), ns)  # noqa: S102
    return ns


def _build_3bus_ybus():
    """3-bus radial line network: bus0(slack)-bus1(PV)-bus2(PQ), r=0.01, x=0.1 each line."""
    r, x = 0.01, 0.1
    y = 1.0 / complex(r, x)
    Y = torch.zeros(3, 3, dtype=torch.complex64)
    for (i, j) in [(0, 1), (1, 2)]:
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y
    return torch.tensor(Y.real, dtype=torch.float32), torch.tensor(Y.imag, dtype=torch.float32)


def test_gradient_flows_into_node_pred_heads():
    ns = _exec_notebook_defs()
    PhysicsConfig = ns["PhysicsConfig"]
    physics_informed_loss_batch = ns["physics_informed_loss_batch"]

    n_buses = 3
    slack_mask = torch.tensor([True, False, False])
    pv_mask = torch.tensor([False, True, False])
    pq_mask = torch.tensor([False, False, True])

    # x layout: [is_slack, is_PV, is_PQ, P, Q, Vmag, Vang]
    x = torch.zeros(n_buses, 7)
    x[0, 0] = 1.0; x[0, 5] = 1.0; x[0, 6] = 0.0        # slack: Vmag=1.0, Vang=0 known
    x[1, 1] = 1.0; x[1, 3] = 0.3; x[1, 5] = 1.02        # PV: P known, Vmag known
    x[2, 2] = 1.0; x[2, 3] = -0.2; x[2, 4] = -0.05      # PQ: P, Q known

    target = torch.zeros(n_buses, 4)  # not used inside the function beyond F.mse_loss(pred, target)

    class _Batch:
        pass

    batch = _Batch()
    batch.batch = torch.zeros(n_buses, dtype=torch.long)
    batch.network_idx = torch.tensor([0])
    batch.x = x
    batch.slack_mask = slack_mask
    batch.pv_mask = pv_mask
    batch.pq_mask = pq_mask

    Y_matrix = _build_3bus_ybus()
    cfg = PhysicsConfig(w_phys=1.0, loss_weight_mode="fixed")

    node_pred = torch.tensor(
        [[1.00, 0.00, 0.05, 0.05],
         [1.02, -0.01, 0.30, 0.02],
         [0.98, -0.02, -0.20, -0.05]],
        requires_grad=True,
    )

    combined_physics, mse_loss, physics_loss, angle_ref_loss, p_res, q_res = physics_informed_loss_batch(
        node_pred, target, batch, networks=[None], Y_cache=[Y_matrix], physics_cfg=cfg,
    )
    combined_physics.backward()

    assert node_pred.grad is not None
    assert node_pred.grad.abs().sum().item() > 0.0
    # gradient must reach all 4 columns (Vmag, Vang, P, Q)
    for col in range(4):
        assert node_pred.grad[:, col].abs().sum().item() > 0.0, f"no gradient reached column {col}"
    print("TEST 1a ok")


if __name__ == "__main__":
    test_edge_local_block_unchanged()
    test_new_nodal_call_present_and_additive()
    test_new_call_never_references_delta_theta_pred()
    test_gated_strictly_on_both_mode_train_loop_only()
    test_gradient_flows_into_node_pred_heads()
    print("\nAll both-mode nodal balance tests passed.")
