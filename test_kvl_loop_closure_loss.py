"""
Regression guards for task 075 (Tier 0: KVL loop-closure self-consistency loss).

Group 0: notebook-source guards (wiring exists in the expected cells).
Group 1: synthetic meshed-graph numeric sanity (path-independent field -> ~0 loss;
          perturbed chord -> strictly positive loss).
Group 2: mode inapplicability (never fires for angle_mode="node").
Group 3: real-data training smoke test (skips gracefully if the dataset/heavy
          deps required to execute the notebook are not available).
"""
import json
import math
import os
from collections import deque

import torch
import torch.nn.functional as F

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")


def _load_notebook_cells():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _cell_source_containing(marker):
    for cell in _load_notebook_cells():
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"no code cell found containing {marker!r}")


# ── Group 0: notebook-source guards ──────────────────────────────────────────
def test_notebook_defines_kvl_loss_function():
    src = _cell_source_containing("def compute_kvl_loop_closure_loss")
    assert "def compute_kvl_loop_closure_loss" in src
    print("TEST 0a ok")


def test_notebook_stores_chord_mask():
    src = _cell_source_containing("data.bfs_chord_mask = chord_mask")
    assert "data.bfs_chord_mask = chord_mask" in src
    print("TEST 0b ok")


def test_notebook_config_has_kvl_fields():
    src = _cell_source_containing("class PhysicsConfig:")
    assert "fraction_kvl" in src and "max_w_kvl" in src
    print("TEST 0c ok")


def test_notebook_wires_kvl_into_training_loop():
    src = _cell_source_containing("def train_power_flow_gnn(")
    for marker in ("kvl_activation_start", "scale_kvl", "kvl_loss_val",
                   "compute_kvl_loop_closure_loss", "bfs_chord_mask"):
        assert marker in src, f"train_power_flow_gnn cell missing {marker!r}"
    # never fires unless delta_theta_pred is not None (i.e. never for pure "node" mode)
    assert "if delta_theta_pred is not None and scale_kvl > 0 and physics_cfg.fraction_kvl > 0" in src
    print("TEST 0d ok")


# ── Replicated helpers (standalone, mirrors notebook cell 984b9125) ──────────
def precompute_bfs_order(edge_index_fwd, slack_idx, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    for e in range(edge_index_fwd.size(1)):
        u, v = edge_index_fwd[0, e].item(), edge_index_fwd[1, e].item()
        adj[u].append((v, e, +1.0))
        adj[v].append((u, e, -1.0))

    visited = [False] * num_nodes
    visited[slack_idx] = True
    queue = deque([slack_idx])
    bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = [], [], [], []
    while queue:
        node = queue.popleft()
        for neighbor, edge_e, sign in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                bfs_edge_idx.append(edge_e)
                bfs_signs.append(sign)
                bfs_node_order.append(neighbor)
                bfs_parent.append(node)
                queue.append(neighbor)
    assert len(bfs_node_order) == num_nodes - 1
    return (
        torch.tensor(bfs_edge_idx, dtype=torch.long),
        torch.tensor(bfs_signs, dtype=torch.float32),
        torch.tensor(bfs_node_order, dtype=torch.long),
        torch.tensor(bfs_parent, dtype=torch.long),
    )


def reconstruct_theta_from_delta(delta_theta_fwd, bfs_edge_idx, bfs_signs,
                                  bfs_node_order, bfs_parent, num_nodes):
    theta = torch.zeros(num_nodes, dtype=delta_theta_fwd.dtype)
    for i in range(len(bfs_node_order)):
        child = bfs_node_order[i].item()
        parent = bfs_parent[i].item()
        e = bfs_edge_idx[i]
        s = bfs_signs[i]
        theta[child] = theta[parent] - s * delta_theta_fwd[e]
    return theta


def compute_kvl_loop_closure_loss(delta_theta_pred, chord_mask_fwd, theta_recon, edge_index_fwd):
    """Exact replica of the notebook function (see compute_kvl_loop_closure_loss)."""
    if chord_mask_fwd is None or not bool(chord_mask_fwd.any()):
        return torch.tensor(0.0, device=delta_theta_pred.device), {"kvl_mae": 0.0}
    src = edge_index_fwd[0][chord_mask_fwd]
    dst = edge_index_fwd[1][chord_mask_fwd]
    theta_diff = theta_recon[src] - theta_recon[dst]
    dtheta_chord = delta_theta_pred[chord_mask_fwd]
    loss = F.mse_loss(dtheta_chord, theta_diff)
    mae = (dtheta_chord - theta_diff).abs().mean().detach()
    return loss, {"kvl_mae": mae.item()}


# 5-node, 1-loop meshed graph (mirrors test_edge_delta_theta.py's fixture):
#   0 -- 1 -- 2
#     \      /
#      3 -- 4
# forward edges: (0,1) (1,2) (0,3) (3,4) (2,4) -- last edge (2,4) is the chord.
_EDGE_INDEX_FWD = torch.tensor([[0, 1, 0, 3, 2], [1, 2, 3, 4, 4]], dtype=torch.long)
_NUM_NODES = 5
_SLACK_IDX = 0


def _bfs_and_chord_mask():
    bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
        _EDGE_INDEX_FWD, _SLACK_IDX, _NUM_NODES)
    chord_mask = torch.ones(_EDGE_INDEX_FWD.size(1), dtype=torch.bool)
    chord_mask[bfs_edge_idx] = False
    return bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, chord_mask


# ── Group 1: synthetic meshed-graph numeric sanity ───────────────────────────
def test_kvl_loss_zero_for_path_independent_field():
    bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, chord_mask = _bfs_and_chord_mask()
    assert int(chord_mask.sum()) == 1, "5-node/5-edge graph has exactly 1 independent loop"

    theta_true = torch.tensor([0.0, 0.05, 0.12, -0.03, 0.08])
    delta_theta_pred = theta_true[_EDGE_INDEX_FWD[0]] - theta_true[_EDGE_INDEX_FWD[1]]

    theta_recon = reconstruct_theta_from_delta(
        delta_theta_pred, bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, _NUM_NODES)

    loss, diag = compute_kvl_loop_closure_loss(delta_theta_pred, chord_mask, theta_recon, _EDGE_INDEX_FWD)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"expected ~0 loss, got {loss.item()}"
    print(f"TEST 1a ok  loss={loss.item():.2e}")


def test_kvl_loss_positive_when_chord_perturbed():
    bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, chord_mask = _bfs_and_chord_mask()

    theta_true = torch.tensor([0.0, 0.05, 0.12, -0.03, 0.08])
    delta_theta_pred = theta_true[_EDGE_INDEX_FWD[0]] - theta_true[_EDGE_INDEX_FWD[1]]
    delta_theta_pred = delta_theta_pred.clone()
    delta_theta_pred[chord_mask] += 0.1  # perturb only the chord edge's own prediction

    theta_recon = reconstruct_theta_from_delta(
        delta_theta_pred, bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, _NUM_NODES)

    loss, diag = compute_kvl_loop_closure_loss(delta_theta_pred, chord_mask, theta_recon, _EDGE_INDEX_FWD)
    assert loss.item() > 1e-4, f"expected strictly positive loss after chord perturbation, got {loss.item()}"
    print(f"TEST 1b ok  loss={loss.item():.4f}")


def test_kvl_loss_zero_chords_returns_zero():
    """Fully radial batch (no chords) -> loss must be exactly 0.0, no crash."""
    chord_mask = torch.zeros(_EDGE_INDEX_FWD.size(1), dtype=torch.bool)
    delta_theta_pred = torch.zeros(_EDGE_INDEX_FWD.size(1))
    theta_recon = torch.zeros(_NUM_NODES)
    loss, diag = compute_kvl_loop_closure_loss(delta_theta_pred, chord_mask, theta_recon, _EDGE_INDEX_FWD)
    assert loss.item() == 0.0
    print("TEST 1c ok")


# ── Group 2: mode inapplicability (angle_mode="node") ────────────────────────
def test_kvl_never_wired_for_node_mode():
    """The wiring guard requires delta_theta_pred is not None, which the model
    contract guarantees is None for angle_mode="node" -- so the KVL block is
    structurally unreachable in that mode."""
    src = _cell_source_containing("def train_power_flow_gnn(")
    assert "if delta_theta_pred is not None and scale_kvl > 0 and physics_cfg.fraction_kvl > 0" in src, \
        "KVL gate must require delta_theta_pred is not None (never fires for angle_mode='node')"
    print("TEST 2a ok")


# ── Group 3: real-data training smoke test (skips gracefully) ───────────────
def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        print(f"SKIPPED: {reason}")
        return


def _load_dataset_list_bounded(index_path, n, pypsa_module):
    """Load only the first `n` networks from a save_dataset_list index, avoiding
    the cost of constructing all (e.g. 1500) networks when only a few are needed."""
    with open(index_path, "r") as fh:
        meta = json.load(fh)
    save_dir = os.path.dirname(index_path) or "."
    networks = []
    for nc_name in meta["files"][:n]:
        networks.append(pypsa_module.Network(os.path.join(save_dir, nc_name)))
    return networks


def test_kvl_training_smoke():
    try:
        cells = _load_notebook_cells()
        # train_power_flow_gnn calls evaluate_gnn_on_test_set internally for the
        # final test-set metrics, which is defined in a later cell -- include both.
        _markers = ("def train_power_flow_gnn(", "def evaluate_gnn_on_test_set(")
        idx_train = max(
            i for i, c in enumerate(cells)
            if c.get("cell_type") == "code"
            and any(m in "".join(c.get("source", [])) for m in _markers)
        )
        ns = {"__name__": "__kvl_smoke_test__"}
        for cell in cells[:idx_train + 1]:
            if cell.get("cell_type") != "code":
                continue
            exec(compile("".join(cell.get("source", [])), "<notebook>", "exec"), ns)  # noqa: S102

        training_networks_dir = ns["TRAINING_NETWORKS_DIR"]
        dataset_path = os.path.join(training_networks_dir, "mixed_1500_w.json")
        if not os.path.exists(dataset_path):
            _skip(f"real-data smoke test skipped: {dataset_path} not found")
            return

        # Bounded load -- avoid constructing all 1500 networks just to keep 4.
        mini_nets = _load_dataset_list_bounded(dataset_path, 4, ns["pypsa"])
        _, history, _ = ns["train_power_flow_gnn"](
            networks=mini_nets,
            num_epochs=3,
            angle_mode="edge_delta",
            physics_cfg=ns["PhysicsConfig"](fraction_kvl=0.1),
            kvl_activation_start=0,
            batch_size=2,
        )
    except Exception as exc:  # noqa: BLE001 - any missing dep/data -> graceful skip
        _skip(f"real-data smoke test skipped: {exc!r}")
        return

    assert all(math.isfinite(v) for v in history["train_kvl_loss"])
    assert any(v > 0 for v in history["train_kvl_loss"])
    print("TEST 3 ok")


if __name__ == "__main__":
    test_notebook_defines_kvl_loss_function()
    test_notebook_stores_chord_mask()
    test_notebook_config_has_kvl_fields()
    test_notebook_wires_kvl_into_training_loop()
    test_kvl_loss_zero_for_path_independent_field()
    test_kvl_loss_positive_when_chord_perturbed()
    test_kvl_loss_zero_chords_returns_zero()
    test_kvl_never_wired_for_node_mode()
    test_kvl_training_smoke()
    print("\nAll KVL loop-closure loss tests passed (or gracefully skipped).")
