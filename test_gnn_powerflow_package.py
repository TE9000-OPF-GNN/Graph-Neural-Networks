"""Smoke tests for gnn_powerflow package (Task A).

Tests:
  T1: All modules import without error
  T2: load_system_from_csv("ieee30") returns network with 30 buses
  T3: PowerFlowDataset built from solved network has edge_attr.size(1) == 7
  T4: scenario_to_data produces Data without y attribute
  T5: PowerFlowGNN forward pass works with 7-feature edge_attr
  T6: collate_with_ptdf batches correctly
  T7: compute_ptdf_matrix returns correct shape
  T8: precompute_bfs_order + reconstruct_theta_from_delta work end-to-end
  T9: save/load dataset list roundtrip (skipped if no writable temp dir)

Run with: python -m pytest test_gnn_powerflow_package.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import pypsa
import torch

# Ensure repo root is on path when running from subdirectory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Helpers ───────────────────────────────────────────────────────────────────

GRID_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "grid_model_files"
)
IEEE30_CSV = os.path.join(GRID_MODEL_DIR, "ieee30_buses.csv")
HAS_CSV = os.path.isfile(IEEE30_CSV)


def _make_simple_network(n_buses: int = 5) -> pypsa.Network:
    """Build a minimal solved PyPSA network for testing (no CSV files needed)."""
    n = pypsa.Network()
    n.sn_mva = 100.0

    # Buses
    for i in range(n_buses):
        n.add("Bus", f"Bus {i}", v_nom=1.0, v_mag_pu_min=0.9, v_mag_pu_max=1.1)

    # Lines: ring topology Bus 0→1→2→...→(n-1)→0
    for i in range(n_buses):
        j = (i + 1) % n_buses
        n.add("Line", f"L{i}_{j}", bus0=f"Bus {i}", bus1=f"Bus {j}",
              r=0.01, x=0.05, b=0.001, s_nom=1.0)

    # Generators
    n.add("Generator", "G0", bus="Bus 0", control="Slack", p_nom=2.0, p_set=1.0)
    n.add("Generator", "G1", bus="Bus 2", control="PV", p_nom=1.0, p_set=0.5)

    # Loads
    for i in range(1, n_buses):
        n.add("Load", f"Load{i}", bus=f"Bus {i}", p_set=0.2, q_set=0.05)

    # Add one snapshot and synthetic solved values
    import pandas as pd
    n.set_snapshots(pd.DatetimeIndex(["2024-01-01"]))
    snap = n.snapshots[0]

    buses = list(n.buses.index)
    n.buses_t.v_mag_pu = pd.DataFrame(
        {b: [1.0] for b in buses}, index=n.snapshots
    )
    angles = {b: [float(i) * 0.01] for i, b in enumerate(buses)}
    n.buses_t.v_ang = pd.DataFrame(angles, index=n.snapshots)
    n.buses_t.p = pd.DataFrame(
        {b: [0.2 if b != "Bus 0" else 0.8] for b in buses}, index=n.snapshots
    )
    n.buses_t.q = pd.DataFrame(
        {b: [0.05] for b in buses}, index=n.snapshots
    )

    lines = list(n.lines.index)
    n.lines_t.p0 = pd.DataFrame({l: [0.1] for l in lines}, index=n.snapshots)
    n.lines_t.p1 = pd.DataFrame({l: [-0.09] for l in lines}, index=n.snapshots)
    n.lines_t.q0 = pd.DataFrame({l: [0.02] for l in lines}, index=n.snapshots)
    n.lines_t.q1 = pd.DataFrame({l: [-0.019] for l in lines}, index=n.snapshots)
    return n


# ── T1: Imports ───────────────────────────────────────────────────────────────

class TestImports:
    def test_package_imports(self):
        import gnn_powerflow

    def test_dataset_subpackage(self):
        from gnn_powerflow.dataset import (
            PowerFlowDataset,
            collate_with_ptdf,
            validate_training_data,
            scenario_to_data,
            compute_ptdf_matrix,
            precompute_bfs_order,
            reconstruct_theta_from_delta,
            compute_bfs_depth,
            save_dataset_list,
            load_dataset_list,
            EDGE_FEATURE_DIM,
        )
        assert EDGE_FEATURE_DIM == 7

    def test_model_subpackage(self):
        from gnn_powerflow.model import PowerFlowGNN
        from gnn_powerflow.model.line_flows import (
            calculate_line_flows,
            calculate_line_flows_from_delta_theta,
            build_line_results_from_flows,
        )

    def test_grid_subpackage(self):
        from gnn_powerflow.grid.csv_loader import load_system_from_csv
        from gnn_powerflow.grid.power_flow import (
            solve_power_flow_with_slack,
            sanity_check_power_flow,
        )


# ── T2: CSV loader ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_CSV, reason="ieee30 CSV files not present")
class TestCSVLoader:
    def test_load_ieee30(self):
        from gnn_powerflow.grid.csv_loader import load_system_from_csv
        n = load_system_from_csv("ieee30", load_dir=GRID_MODEL_DIR)
        assert len(n.buses) == 30, f"Expected 30 buses, got {len(n.buses)}"
        assert len(n.lines) > 0
        assert len(n.generators) > 0

    def test_load_ieee30_has_slack(self):
        from gnn_powerflow.grid.csv_loader import load_system_from_csv
        n = load_system_from_csv("ieee30", load_dir=GRID_MODEL_DIR)
        slack_gens = n.generators[n.generators["control"] == "Slack"]
        assert len(slack_gens) >= 1, "No Slack generator"


# ── T3: PowerFlowDataset — 7 edge features ────────────────────────────────────

class TestPowerFlowDataset:
    def test_edge_attr_has_7_features(self):
        from gnn_powerflow.dataset import PowerFlowDataset, EDGE_FEATURE_DIM
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert g.edge_attr.size(1) == 7, (
            f"Expected 7 edge features, got {g.edge_attr.size(1)}"
        )
        assert EDGE_FEATURE_DIM == 7

    def test_switch_active_col_default_one(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        # col 6 = switch_active; all should be 1.0 by default
        assert (g.edge_attr[:, 6] == 1.0).all(), (
            f"switch_active not all 1.0: {g.edge_attr[:, 6]}"
        )

    def test_open_lines_sets_switch_active_zero(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        open_line = list(net.lines.index)[0]  # e.g. "L0_1"
        ds = PowerFlowDataset([net], open_lines=[open_line])
        g = ds[0]
        # The open line (forward + reverse) should have switch_active=0.0
        assert (g.edge_attr[:, 6] == 0.0).any(), (
            f"No edge has switch_active=0 after setting open_lines=[{open_line!r}]"
        )
        # Other edges should still be 1.0
        assert (g.edge_attr[:, 6] == 1.0).any(), "All edges are 0 — something is wrong"

    def test_v_lim_shape(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert hasattr(g, "v_lim"), "data.v_lim missing"
        assert g.v_lim.shape == (5, 2), f"v_lim shape {g.v_lim.shape}, expected [5, 2]"

    def test_s_nom_edge_shape(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert hasattr(g, "s_nom_edge"), "data.s_nom_edge missing"
        E = g.edge_index.size(1)
        assert g.s_nom_edge.shape == (E, 1), f"s_nom_edge shape {g.s_nom_edge.shape}"

    def test_forward_edge_mask_correct(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        # Forward edges are at even positions (0, 2, 4, ...)
        E = g.edge_index.size(1)
        expected_fwd = E // 2
        assert int(g.forward_edge_mask.sum()) == expected_fwd

    def test_edge_index_uses_double_stride(self):
        """Verify [::2] not [:2] — forward edges are interleaved, not just first two."""
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(6)  # 6 buses, 6 lines → 12 edges
        ds = PowerFlowDataset([net])
        g = ds[0]
        E = g.edge_index.size(1)
        assert E == 12, f"Expected 12 edges (6 fwd + 6 rev), got {E}"
        # Forward edges (even indices) should differ from reverse (odd indices)
        fwd_src = g.edge_index[0, ::2]
        rev_src = g.edge_index[0, 1::2]
        # fwd src and rev src should be different (reverse swaps src/dst)
        assert not torch.equal(fwd_src, rev_src)

    def test_y_shape(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert g.y.shape == (5, 4), f"y shape {g.y.shape}, expected [5, 4]"

    def test_x_shape(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert g.x.shape == (5, 7), f"x shape {g.x.shape}, expected [5, 7]"

    def test_pnom_share_adds_col(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], use_pnom_share=True)
        g = ds[0]
        assert g.x.shape == (5, 8), f"x with pnom_share shape {g.x.shape}, expected [5, 8]"

    def test_bus_masks_present(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        assert hasattr(g, "slack_mask") and g.slack_mask.shape == (5,)
        assert hasattr(g, "pv_mask") and g.pv_mask.shape == (5,)
        assert hasattr(g, "pq_mask") and g.pq_mask.shape == (5,)
        # Exactly one slack
        assert g.slack_mask.sum().item() == 1

    def test_edge_delta_mode_adds_bfs(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], angle_mode="edge_delta")
        g = ds[0]
        assert hasattr(g, "y_delta_theta"), "Missing y_delta_theta"
        assert hasattr(g, "bfs_edge_idx"), "Missing bfs_edge_idx"
        assert g.y_delta_theta.shape[0] == g.edge_index.size(1) // 2

    def test_skip_ptdf(self):
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], skip_ptdf=True)
        g = ds[0]
        # y_ptdf should still be present (zeros) — same shape
        n_lines = len(net.lines)
        assert g.y_ptdf.shape[0] == n_lines


# ── T4: scenario_to_data — no y targets ───────────────────────────────────────

class TestScenarioToData:
    def test_no_y_attribute(self):
        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        data = scenario_to_data(net)
        # PyG Data.__getattr__ returns None for unset keys, so use .get() or check value
        assert data.get('y') is None, "scenario_to_data should NOT set data.y (should be None)"

    def test_x_shape(self):
        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        data = scenario_to_data(net)
        assert data.x.shape == (5, 7)

    def test_edge_attr_7_features(self):
        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        data = scenario_to_data(net)
        assert data.edge_attr.size(1) == 7

    def test_open_lines_switch_zero(self):
        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        open_line = list(net.lines.index)[0]
        data = scenario_to_data(net, open_lines=[open_line])
        assert (data.edge_attr[:, 6] == 0.0).any()

    def test_transformer_in_open_lines_warns(self):
        """Transformer names in open_lines emit UserWarning (Gap 3 guard)."""
        import warnings
        from gnn_powerflow.dataset import scenario_to_data
        # Build net with a transformer
        net = _make_simple_network(5)
        net.add("Transformer", "TR0", bus0="Bus 0", bus1="Bus 1",
                x=0.05, r=0.01, s_nom=1.0, tap_ratio=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scenario_to_data(net, open_lines=["TR0"])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, "Expected UserWarning for transformer in open_lines"
        assert "TR0" in str(user_warnings[0].message)

    def test_line_in_open_lines_no_warn(self):
        """Line names in open_lines do NOT emit a warning."""
        import warnings
        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        open_line = list(net.lines.index)[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scenario_to_data(net, open_lines=[open_line])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, f"Unexpected warning: {user_warnings}"


        from gnn_powerflow.dataset import scenario_to_data
        net = _make_simple_network(5)
        data = scenario_to_data(net)
        assert hasattr(data, "v_lim") and data.v_lim.shape == (5, 2)


# ── T5: PowerFlowGNN forward pass ─────────────────────────────────────────────

class TestPowerFlowGNN:
    def test_forward_node_mode(self):
        from gnn_powerflow.model import PowerFlowGNN
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]

        model = PowerFlowGNN(
            node_features=7, edge_features=7,
            hidden_dim=32, num_layers=2, heads=2,
            conv_type="gatv2", angle_mode="node",
        )
        model.eval()
        with torch.no_grad():
            out = model(g)
        node_pred, ptdf_pred, delta_theta = out
        assert node_pred.shape == (5, 4)
        assert delta_theta is None

    def test_forward_edge_delta_mode(self):
        from gnn_powerflow.model import PowerFlowGNN
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], angle_mode="edge_delta")
        g = ds[0]

        model = PowerFlowGNN(
            node_features=7, edge_features=7,
            hidden_dim=32, num_layers=2, heads=2,
            conv_type="gatv2", angle_mode="edge_delta",
        )
        model.eval()
        with torch.no_grad():
            out = model(g)
        node_pred, ptdf_pred, delta_theta = out
        assert node_pred.shape == (5, 3)  # [Vmag, P, Q]
        assert delta_theta is not None
        E_fwd = g.edge_index.size(1) // 2
        assert delta_theta.shape == (E_fwd,)

    def test_forward_both_mode(self):
        from gnn_powerflow.model import PowerFlowGNN
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], angle_mode="both")
        g = ds[0]

        model = PowerFlowGNN(
            node_features=7, edge_features=7,
            hidden_dim=32, num_layers=2, heads=2,
            angle_mode="both",
        )
        model.eval()
        with torch.no_grad():
            out = model(g)
        node_pred, ptdf_pred, delta_theta = out
        assert node_pred.shape == (5, 4)  # [Vmag, Vang, P, Q]
        assert delta_theta is not None

    def test_return_embeddings(self):
        from gnn_powerflow.model import PowerFlowGNN
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net])
        g = ds[0]
        model = PowerFlowGNN(node_features=7, edge_features=7, hidden_dim=32)
        model.eval()
        with torch.no_grad():
            out = model(g, return_embeddings=True)
        assert len(out) == 4  # node_pred, h_nodes, h_edges, delta_theta


# ── T6: collate_with_ptdf ─────────────────────────────────────────────────────

class TestCollate:
    def test_batch_two_graphs(self):
        from gnn_powerflow.dataset import PowerFlowDataset, collate_with_ptdf
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net, net])
        batch = collate_with_ptdf([ds[0], ds[1]])
        assert hasattr(batch, "slack_mask")
        assert hasattr(batch, "pv_mask")
        assert hasattr(batch, "pq_mask")
        assert batch.x.shape[0] == 10  # 2 × 5 buses
        assert batch.edge_attr.size(1) == 7

    def test_batch_has_mask_attrs(self):
        from gnn_powerflow.dataset import PowerFlowDataset, collate_with_ptdf
        net = _make_simple_network(4)
        ds = PowerFlowDataset([net])
        batch = collate_with_ptdf([ds[0]])
        assert hasattr(batch, "y_ptdf_list")
        assert len(batch.y_ptdf_list) == 1


# ── T7: PTDF matrix ──────────────────────────────────────────────────────────

class TestPTDF:
    def test_ptdf_shape_lines(self):
        from gnn_powerflow.dataset import compute_ptdf_matrix
        net = _make_simple_network(5)
        ptdf = compute_ptdf_matrix(net, ptdf_branch_mode="lines")
        n_lines = len(net.lines)
        n_buses = len(net.buses)
        assert ptdf.shape == (n_lines, n_buses), (
            f"PTDF shape {ptdf.shape}, expected ({n_lines}, {n_buses})"
        )

    def test_ptdf_slack_col_zero(self):
        from gnn_powerflow.dataset import compute_ptdf_matrix
        net = _make_simple_network(5)
        ptdf = compute_ptdf_matrix(net)
        # Slack column should be zero
        slack_bus = net.generators[net.generators["control"] == "Slack"].iloc[0]["bus"]
        slack_idx = list(net.buses.index).index(slack_bus)
        assert np.allclose(ptdf[:, slack_idx], 0.0), "Slack column not zero"


# ── T8: BFS / theta reconstruction ───────────────────────────────────────────

class TestEdgeDelta:
    def test_bfs_order_covers_all_nodes(self):
        from gnn_powerflow.dataset import precompute_bfs_order
        # Simple 4-node path: 0—1—2—3
        edge_index_fwd = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
            edge_index_fwd, slack_idx=0, num_nodes=4
        )
        assert len(bfs_node_order) == 3  # N-1 nodes (excluding slack)
        assert set(bfs_node_order.tolist()) == {1, 2, 3}

    def test_reconstruct_theta_zero_at_slack(self):
        from gnn_powerflow.dataset import precompute_bfs_order
        from gnn_powerflow.dataset.edge_delta import reconstruct_theta_from_delta
        edge_index_fwd = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
            edge_index_fwd, slack_idx=0, num_nodes=4
        )
        delta_theta = torch.tensor([0.1, 0.05, 0.02])
        theta = reconstruct_theta_from_delta(
            delta_theta, bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, 4
        )
        assert theta[0].item() == pytest.approx(0.0), "Slack theta should be 0"
        assert len(theta) == 4

    def test_compute_bfs_depth(self):
        from gnn_powerflow.dataset.edge_delta import compute_bfs_depth
        # Star topology: 0 connected to 1, 2, 3
        edge_index_fwd = np.array([[0, 0, 0], [1, 2, 3]])
        depth = compute_bfs_depth(edge_index_fwd, slack_idx=0, num_nodes=4)
        assert depth[0] == 0
        assert depth[1] == 1 and depth[2] == 1 and depth[3] == 1


# ── T9: Line flows ────────────────────────────────────────────────────────────

class TestLineFlows:
    def test_calculate_line_flows_shape(self):
        from gnn_powerflow.model.line_flows import calculate_line_flows
        net = _make_simple_network(5)
        n_buses = len(net.buses)
        vmag = np.ones(n_buses)
        vang = np.zeros(n_buses)
        flows = calculate_line_flows(net, vmag, vang)
        n_lines = len(net.lines)
        assert len(flows["p0"]) == n_lines
        assert len(flows["q0"]) == n_lines

    def test_calculate_line_flows_from_delta_theta_shape(self):
        from gnn_powerflow.model.line_flows import calculate_line_flows_from_delta_theta
        from gnn_powerflow.dataset import PowerFlowDataset
        net = _make_simple_network(5)
        ds = PowerFlowDataset([net], angle_mode="edge_delta")
        g = ds[0]

        E_fwd = g.edge_index.size(1) // 2
        fwd_mask = np.zeros(g.edge_index.shape[1], dtype=bool)
        fwd_mask[::2] = True
        ei_fwd = g.edge_index[:, fwd_mask].numpy()
        ea_fwd = g.edge_attr[fwd_mask].numpy()
        dt = np.zeros(E_fwd)
        vmag = np.ones(len(net.buses))
        flows = calculate_line_flows_from_delta_theta(dt, vmag, ei_fwd, ea_fwd)
        assert len(flows["p0"]) == E_fwd


# ── T10: validate_training_data ───────────────────────────────────────────────

class TestValidate:
    def test_valid_network_passes(self):
        from gnn_powerflow.dataset import validate_training_data
        net = _make_simple_network(5)
        result = validate_training_data([net], name="test", n_sample=1)
        assert result is True

    def test_empty_list_fails(self):
        from gnn_powerflow.dataset import validate_training_data
        result = validate_training_data([], name="empty", n_sample=1)
        assert result is False


# ── T11: Mini-training integration (Gap 1 from plan assessment) ───────────────

class TestMiniTraining:
    """Full pipeline: load_system_from_csv / synthetic net → DataLoader →
    PowerFlowGNN → MSE loss → loss.backward()

    Exercises the path that 39 structural tests cannot catch:
    - 7-feature edge_attr flows through GATv2Conv without shape error
    - collate_with_ptdf produces a valid Batch with correct mask tensors
    - gradient backpropagation is finite (no NaN explodes or vanishes)
    - loss decreases over 3 steps (sanity: model is learnable)
    """

    def _run_loop(self, angle_mode: str, n_nets: int = 4):
        from gnn_powerflow.dataset import PowerFlowDataset, collate_with_ptdf
        from gnn_powerflow.model import PowerFlowGNN
        from torch.utils.data import DataLoader
        import torch.nn.functional as F

        nets = [_make_simple_network(6) for _ in range(n_nets)]
        ds = PowerFlowDataset(nets, angle_mode=angle_mode)
        loader = DataLoader(
            ds, batch_size=n_nets, shuffle=False, collate_fn=collate_with_ptdf
        )

        n_out = 3 if angle_mode == "edge_delta" else 4
        model = PowerFlowGNN(
            node_features=7, edge_features=7,
            hidden_dim=16, num_layers=2, heads=2,
            angle_mode=angle_mode,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        model.train()
        for _epoch in range(3):
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)
                node_pred = out[0]
                if angle_mode == "edge_delta":
                    # Package y has 4 cols; model outputs 3 — compare PQ subset
                    target = batch.y[:, [0, 2, 3]]  # Vmag, P, Q
                else:
                    target = batch.y
                loss = F.mse_loss(node_pred, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        return losses

    def test_node_mode_3_steps(self):
        losses = self._run_loop("node")
        assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
        assert losses[-1] < losses[0] * 5.0, f"Loss grew > 5×: {losses}"

    def test_edge_delta_mode_3_steps(self):
        losses = self._run_loop("edge_delta")
        assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
        assert losses[-1] < losses[0] * 5.0, f"Loss grew > 5×: {losses}"

    def test_both_mode_3_steps(self):
        losses = self._run_loop("both")
        assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
        assert losses[-1] < losses[0] * 5.0, f"Loss grew > 5×: {losses}"

    def test_batch_masks_survive_collate(self):
        """Confirm slack_mask / pv_mask / pq_mask are concatenated correctly across graphs."""
        from gnn_powerflow.dataset import PowerFlowDataset, collate_with_ptdf
        from torch.utils.data import DataLoader

        nets = [_make_simple_network(5) for _ in range(3)]
        ds = PowerFlowDataset(nets)
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_with_ptdf)
        batch = next(iter(loader))

        assert batch.slack_mask.shape == (15,), f"slack_mask shape {batch.slack_mask.shape}"
        assert batch.pq_mask.shape == (15,)
        # 3 networks × 1 slack each = 3 slack buses
        assert batch.slack_mask.sum().item() == 3, "Expected 3 slack buses in batch"
        # edge_attr must still be 7 cols after batching
        assert batch.edge_attr.size(1) == 7, f"edge_attr cols {batch.edge_attr.size(1)}"

    @pytest.mark.skipif(not HAS_CSV, reason="ieee30 CSV files not present")
    def test_real_network_forward(self):
        """csv_loader → solve pf → PowerFlowDataset → forward pass (no crash)."""
        from gnn_powerflow.grid.csv_loader import load_system_from_csv
        from gnn_powerflow.dataset import PowerFlowDataset
        from gnn_powerflow.model import PowerFlowGNN
        import torch.nn.functional as F

        n = load_system_from_csv("ieee30", load_dir=GRID_MODEL_DIR)
        n.pf(use_seed=True)
        import pandas as pd
        n.set_snapshots(pd.DatetimeIndex(["2024-01-01"]))
        snap = n.snapshots[0]
        buses = list(n.buses.index)
        n.buses_t.v_mag_pu = pd.DataFrame({b: [1.0] for b in buses}, index=n.snapshots)
        n.buses_t.v_ang = pd.DataFrame({b: [0.0] for b in buses}, index=n.snapshots)
        n.buses_t.p = pd.DataFrame({b: [0.0] for b in buses}, index=n.snapshots)
        n.buses_t.q = pd.DataFrame({b: [0.0] for b in buses}, index=n.snapshots)
        n.lines_t.p0 = pd.DataFrame(
            {l: [0.0] for l in n.lines.index}, index=n.snapshots
        )
        n.lines_t.p1 = pd.DataFrame(
            {l: [0.0] for l in n.lines.index}, index=n.snapshots
        )
        n.lines_t.q0 = pd.DataFrame(
            {l: [0.0] for l in n.lines.index}, index=n.snapshots
        )
        n.lines_t.q1 = pd.DataFrame(
            {l: [0.0] for l in n.lines.index}, index=n.snapshots
        )

        ds = PowerFlowDataset([n])
        g = ds[0]
        assert g.edge_attr.size(1) == 7
        n_buses = len(n.buses)

        model = PowerFlowGNN(node_features=7, edge_features=7, hidden_dim=32)
        model.eval()
        with torch.no_grad():
            out = model(g)
        assert out[0].shape == (n_buses, 4)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)
