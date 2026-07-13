"""PyG dataset classes for GNN power-flow training and simulation inference.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 12 /
        GNN_Powerflow_V2.7_Analysis.ipynb (PowerFlowDataset cell)
TODO: remove duplicate inline definitions from notebooks once refactored.

Key differences from the notebook version:
- edge_attr has 7 columns: [r, x, b/2, tap, g_ser, b_ser, switch_active]
  col 6 = switch_active: 1.0 = line in service, 0.0 = open/contingency
- data.v_lim [N,2]: per-bus [v_min_pu, v_max_pu] from network.buses
- data.s_nom_edge [E,1]: per-edge thermal rating (s_nom column)
- _create_graph_data gains open_lines and skip_ptdf parameters
- scenario_to_data provides a PF-free inference path (no y targets)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pypsa
import torch
from torch_geometric.data import Batch, Data, Dataset

from gnn_powerflow.dataset.edge_delta import precompute_bfs_order
from gnn_powerflow.dataset.ptdf import compute_ptdf_matrix

logger = logging.getLogger(__name__)

# Number of edge features in the package version (7)
EDGE_FEATURE_DIM = 7


def _build_graph_edges(
    network: pypsa.Network,
    open_lines: list[str] | None = None,
    use_edge_features: bool = True,
) -> tuple[
    torch.Tensor,     # edge_index [2, 2E]
    torch.Tensor,     # edge_attr  [2E, 7]
    torch.Tensor,     # forward_edge_mask [2E] bool
    torch.Tensor,     # dc_flow_mask      [2E] bool
    torch.Tensor,     # s_nom_edge        [2E, 1]
    list[int],        # fwd_line_idx_list (PTDF supervision indices)
    list[int],        # ptdf_edge_row_idx_list (per-edge PTDF row idx or -1)
]:
    """
    Build bidirectional edge tensors from lines + transformers.

    open_lines: line names to mark as switch_active=0.0 (contingency encoding).
                Edges are still present in edge_index; only col 6 changes.
    """
    buses = list(network.buses.index)
    bus_to_i = {b: i for i, b in enumerate(buses)}
    open_set = set(open_lines) if open_lines else set()

    edge_index_list: list[list[int]] = []
    edge_attr_list: list[list[float]] = []
    forward_edge_mask: list[bool] = []
    dc_flow_mask: list[bool] = []
    s_nom_edge_list: list[float] = []
    fwd_line_idx_list: list[int] = []
    ptdf_edge_row_idx_list: list[int] = []

    branches: list[tuple] = []

    # Warn if any transformer name was passed in open_lines — they are silently
    # ignored (switch_active gate only checks btype == "line").
    if open_lines:
        open_set = set(open_lines)
        trafo_names = set(network.transformers.index)
        trafo_in_open = open_set & trafo_names
        if trafo_in_open:
            import warnings
            warnings.warn(
                f"open_lines contains transformer name(s) {sorted(trafo_in_open)}. "
                f"Transformer switch_active is NOT set to 0.0 — only lines are "
                f"supported. Remove transformer names from open_lines.",
                UserWarning,
                stacklevel=3,
            )
    for line_name, line in network.lines.iterrows():
        if line["bus0"] in bus_to_i and line["bus1"] in bus_to_i:
            branches.append((
                line_name, line["bus0"], line["bus1"],
                line["r"], line["x"], line.get("b", 0.0), 1.0,
                "line", float(line.get("s_nom", 1.0)),
            ))

    # Transformers
    for tr_name, tr in network.transformers.iterrows():
        if tr["bus0"] in bus_to_i and tr["bus1"] in bus_to_i:
            tap = tr.get("tap_ratio", 1.0)
            tap = tap if not pd.isna(tap) else 1.0
            branches.append((
                tr_name, tr["bus0"], tr["bus1"],
                tr["r"], tr["x"], tr.get("b", 0.0), float(tap),
                "trafo", float(tr.get("s_nom", 1.0)),
            ))

    for branch_idx, (name, b0, b1, r, x_val, b_val, tap, btype, s_nom) in enumerate(branches):
        i0, i1 = bus_to_i[b0], bus_to_i[b1]
        z = complex(r, x_val)
        y_ser = (1.0 / z) if abs(z) > 1e-12 else 0.0
        g_ser, b_ser = y_ser.real, y_ser.imag

        # switch_active: 0.0 for open lines, 1.0 otherwise
        switch_active = 0.0 if (btype == "line" and name in open_set) else 1.0

        if use_edge_features:
            attr = [r, x_val, b_val / 2.0, tap, g_ser, b_ser, switch_active]
        else:
            attr = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, switch_active]

        is_supervised = (btype == "line")

        # Forward edge
        edge_index_list.append([i0, i1])
        edge_attr_list.append(attr)
        forward_edge_mask.append(is_supervised)
        dc_flow_mask.append(btype == "line")
        s_nom_edge_list.append(s_nom)
        if is_supervised:
            fwd_line_idx_list.append(branch_idx)
            ptdf_edge_row_idx_list.append(branch_idx)
        else:
            ptdf_edge_row_idx_list.append(-1)

        # Reverse edge
        edge_index_list.append([i1, i0])
        edge_attr_list.append(attr)
        forward_edge_mask.append(False)
        dc_flow_mask.append(False)
        s_nom_edge_list.append(s_nom)
        ptdf_edge_row_idx_list.append(-1)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        fwd_mask = torch.tensor(forward_edge_mask, dtype=torch.bool)
        dc_fwd_mask = torch.tensor(dc_flow_mask, dtype=torch.bool)
        s_nom_edge = torch.tensor(s_nom_edge_list, dtype=torch.float).unsqueeze(1)
        line_idx = torch.tensor(fwd_line_idx_list, dtype=torch.long)
        ptdf_edge_row_idx = torch.tensor(ptdf_edge_row_idx_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float)
        fwd_mask = torch.zeros(0, dtype=torch.bool)
        dc_fwd_mask = torch.zeros(0, dtype=torch.bool)
        s_nom_edge = torch.zeros((0, 1), dtype=torch.float)
        line_idx = torch.zeros(0, dtype=torch.long)
        ptdf_edge_row_idx = torch.zeros(0, dtype=torch.long)

    return (
        edge_index, edge_attr, fwd_mask, dc_fwd_mask,
        s_nom_edge, fwd_line_idx_list, ptdf_edge_row_idx_list,
    )


class PowerFlowDataset(Dataset):
    """
    One data object per (network, snapshot) pair — 7-feature edge version.

    edge_attr columns: [r, x, b/2, tap, g_ser, b_ser, switch_active]
    Added vs notebook: data.v_lim [N,2], data.s_nom_edge [E,1]

    Source: GNN_Powerflow_V2.7_Training.ipynb cell 12
    TODO: remove duplicate from notebook once refactored.
    """

    def __init__(
        self,
        networks: list,
        use_edge_features: bool = True,
        use_pnet_balance: bool = True,   # kept for backward compat (ignored)
        use_pnom_share: bool = False,
        transform=None,
        pre_transform=None,
        ptdf_branch_mode: str = "lines",
        angle_mode: str = "node",
        open_lines: list[str] | None = None,
        skip_ptdf: bool = False,
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.networks = networks
        self.use_edge_features = use_edge_features
        self.use_pnom_share = use_pnom_share
        self.ptdf_branch_mode = ptdf_branch_mode
        self.angle_mode = angle_mode
        self.open_lines = open_lines
        self.skip_ptdf = skip_ptdf

        self._index: list[tuple[int, int]] = []
        for net_idx, net in enumerate(networks):
            for t_idx in range(len(net.snapshots)):
                self._index.append((net_idx, t_idx))

    def len(self) -> int:
        return len(self._index)

    def get(self, idx: int) -> Data:
        net_idx, t_idx = self._index[idx]
        network = self.networks[net_idx]
        return _create_graph_data(
            network, t_idx, net_idx,
            open_lines=self.open_lines,
            use_edge_features=self.use_edge_features,
            use_pnom_share=self.use_pnom_share,
            ptdf_branch_mode=self.ptdf_branch_mode,
            angle_mode=self.angle_mode,
            skip_ptdf=self.skip_ptdf,
        )


def _create_graph_data(
    network: pypsa.Network,
    t_idx: int,
    net_idx: int,
    open_lines: list[str] | None = None,
    use_edge_features: bool = True,
    use_pnom_share: bool = False,
    ptdf_branch_mode: str = "lines",
    angle_mode: str = "node",
    skip_ptdf: bool = False,
) -> Data:
    """
    Build a PyG Data object from a solved PyPSA network snapshot.

    7-feature edge_attr: [r, x, b/2, tap, g_ser, b_ser, switch_active]
    Added vs notebook version:
      - data.v_lim     [N, 2]: [v_mag_pu_min, v_mag_pu_max] per bus
      - data.s_nom_edge [E, 1]: per-edge thermal rating
      - open_lines:    line names to mark switch_active=0.0
      - skip_ptdf:     skip PTDF computation (e.g. for contingency scenarios)

    Source: GNN_Powerflow_V2.7_Training.ipynb cell 12 (6-feature version)
    TODO: remove duplicate from notebook once refactored.
    """
    buses = list(network.buses.index)
    n_buses = len(buses)
    bus_to_i = {b: i for i, b in enumerate(buses)}
    snapshot = network.snapshots[t_idx]

    # ── Bus type flags ────────────────────────────────────────────────────────
    is_slack = torch.zeros(n_buses, dtype=torch.float)
    is_pv = torch.zeros(n_buses, dtype=torch.float)
    is_pq = torch.zeros(n_buses, dtype=torch.float)
    slack_mask = torch.zeros(n_buses, dtype=torch.bool)
    pv_mask = torch.zeros(n_buses, dtype=torch.bool)
    pq_mask = torch.zeros(n_buses, dtype=torch.bool)

    for gen_name, gen in network.generators.iterrows():
        bus_name = gen["bus"]
        if bus_name not in bus_to_i:
            continue
        i = bus_to_i[bus_name]
        ctrl = gen["control"]
        if ctrl == "Slack":
            is_slack[i] = 1.0
            slack_mask[i] = True
        elif ctrl == "PV":
            is_pv[i] = 1.0
            pv_mask[i] = True

    for i in range(n_buses):
        if not slack_mask[i] and not pv_mask[i]:
            is_pq[i] = 1.0
            pq_mask[i] = True

    # ── Bus injections / voltages from solved PF ──────────────────────────────
    p_bus = torch.tensor(
        network.buses_t.p.loc[snapshot, buses].values, dtype=torch.float
    )
    q_bus = torch.tensor(
        network.buses_t.q.loc[snapshot, buses].values, dtype=torch.float
    )
    v_mag = torch.tensor(
        network.buses_t.v_mag_pu.loc[snapshot, buses].values, dtype=torch.float
    )
    v_ang = torch.tensor(
        network.buses_t.v_ang.loc[snapshot, buses].values, dtype=torch.float
    )

    # ── Input features (mask unknowns to zero) ────────────────────────────────
    x_p = p_bus.clone()
    x_q = q_bus.clone()
    x_vmag = v_mag.clone()
    x_vang = v_ang.clone()

    x_p[slack_mask] = 0.0
    x_q[slack_mask] = 0.0
    x_q[pv_mask] = 0.0
    x_vmag[pq_mask] = 0.0
    x_vang[pq_mask] = 0.0
    x_vang[pv_mask] = 0.0

    x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)

    if use_pnom_share:
        p_nom_share_col = torch.zeros(n_buses, 1)
        p_noms = network.generators["p_nom"].values.astype(float)
        total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
        for gen_name, gen in network.generators.iterrows():
            b = gen["bus"]
            if b in bus_to_i:
                p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
        x = torch.cat([x, p_nom_share_col], dim=1)

    # ── Target ───────────────────────────────────────────────────────────────
    y = torch.stack([v_mag, v_ang, p_bus, q_bus], dim=1)

    # ── Voltage limits ────────────────────────────────────────────────────────
    v_min_col = "v_mag_pu_min"
    v_max_col = "v_mag_pu_max"
    v_min_vals = (
        network.buses[v_min_col].values.astype(float)
        if v_min_col in network.buses.columns
        else np.full(n_buses, 0.9)
    )
    v_max_vals = (
        network.buses[v_max_col].values.astype(float)
        if v_max_col in network.buses.columns
        else np.full(n_buses, 1.1)
    )
    v_lim = torch.tensor(
        np.column_stack([v_min_vals, v_max_vals]), dtype=torch.float
    )  # [N, 2]

    # ── Edges ─────────────────────────────────────────────────────────────────
    (
        edge_index, edge_attr, fwd_mask, dc_fwd_mask,
        s_nom_edge, fwd_line_idx_list, ptdf_edge_row_idx_list,
    ) = _build_graph_edges(network, open_lines=open_lines, use_edge_features=use_edge_features)

    line_idx = torch.tensor(fwd_line_idx_list, dtype=torch.long)
    ptdf_edge_row_idx = torch.tensor(ptdf_edge_row_idx_list, dtype=torch.long)

    assert edge_attr.size(1) == EDGE_FEATURE_DIM, (
        f"Expected {EDGE_FEATURE_DIM} edge features, got {edge_attr.size(1)}"
    )

    # ── PTDF ──────────────────────────────────────────────────────────────────
    if skip_ptdf:
        n_lines = len(network.lines)
        n_branches = (
            n_lines + len(network.transformers)
            if ptdf_branch_mode == "all"
            else n_lines
        )
        y_ptdf = torch.zeros((n_branches, n_buses), dtype=torch.float)
    else:
        try:
            ptdf_matrix = compute_ptdf_matrix(network, ptdf_branch_mode=ptdf_branch_mode)
            y_ptdf = torch.tensor(ptdf_matrix, dtype=torch.float)
        except Exception:
            n_lines = len(network.lines)
            n_branches = (
                n_lines + len(network.transformers)
                if ptdf_branch_mode == "all"
                else n_lines
            )
            y_ptdf = torch.zeros((n_branches, n_buses), dtype=torch.float)

    # PTDF shape assertion
    n_lines = len(network.lines)
    expected_ptdf_rows = (
        n_lines + len(network.transformers)
        if ptdf_branch_mode == "all"
        else n_lines
    )
    assert y_ptdf.shape[0] == expected_ptdf_rows, (
        f"y_ptdf rows {y_ptdf.shape[0]} != expected {expected_ptdf_rows}"
    )

    # ── Line flows ────────────────────────────────────────────────────────────
    if not network.lines_t.p0.empty:
        y_line_p = torch.tensor(
            network.lines_t.p0.loc[snapshot].values, dtype=torch.float
        )
    else:
        y_line_p = torch.zeros(len(network.lines), dtype=torch.float)

    if ptdf_branch_mode == "all" and len(network.transformers) > 0:
        if not network.transformers_t.p0.empty:
            y_trafo_p = torch.tensor(
                network.transformers_t.p0.loc[snapshot].values, dtype=torch.float
            )
        else:
            y_trafo_p = torch.zeros(len(network.transformers), dtype=torch.float)
        y_line_p = torch.cat([y_line_p, y_trafo_p])

    if not network.lines_t.q0.empty:
        y_line_q = torch.tensor(
            network.lines_t.q0.loc[snapshot].values, dtype=torch.float
        )
    else:
        y_line_q = torch.zeros(len(network.lines), dtype=torch.float)

    if ptdf_branch_mode == "all" and len(network.transformers) > 0:
        if not network.transformers_t.q0.empty:
            y_trafo_q = torch.tensor(
                network.transformers_t.q0.loc[snapshot].values, dtype=torch.float
            )
        else:
            y_trafo_q = torch.zeros(len(network.transformers), dtype=torch.float)
        y_line_q = torch.cat([y_line_q, y_trafo_q])

    # ── Diagonal admittance ───────────────────────────────────────────────────
    if edge_index.size(1) > 0 and edge_attr.size(0) > 0:
        from torch_geometric.utils import scatter
        _src_nodes = edge_index[0]
        _g_ser_all = edge_attr[:, 4]
        _b_ser_all = edge_attr[:, 5]
        _b_half_all = edge_attr[:, 2]
        g_diag = scatter(_g_ser_all, _src_nodes, dim=0, dim_size=n_buses, reduce="sum")
        b_diag = scatter(
            _b_ser_all + _b_half_all, _src_nodes, dim=0, dim_size=n_buses, reduce="sum"
        )
    else:
        g_diag = torch.zeros(n_buses, dtype=torch.float)
        b_diag = torch.zeros(n_buses, dtype=torch.float)

    # ── Assemble Data ─────────────────────────────────────────────────────────
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.slack_mask = slack_mask
    data.pv_mask = pv_mask
    data.pq_mask = pq_mask
    data.y_ptdf = y_ptdf
    data.y_line_p = y_line_p
    data.y_line_q = y_line_q
    data.forward_edge_mask = fwd_mask
    data.dc_flow_mask = dc_fwd_mask
    data.g_diag = g_diag
    data.b_diag = b_diag
    data.ptdf_line_index = line_idx
    data.ptdf_edge_row_idx = ptdf_edge_row_idx
    data.s_nom_edge = s_nom_edge        # [E, 1] — new in package version
    data.v_lim = v_lim                  # [N, 2] — new in package version
    data.network_idx = torch.tensor([net_idx], dtype=torch.long)

    # ── Δθ targets and BFS metadata ───────────────────────────────────────────
    if angle_mode in ("edge_delta", "both"):
        all_fwd_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        all_fwd_mask[::2] = True  # MANDATORY: [::2] not [:2]
        edge_index_fwd = edge_index[:, all_fwd_mask]
        theta_true = y[:, 1]
        y_delta_theta = theta_true[edge_index_fwd[0]] - theta_true[edge_index_fwd[1]]
        data.y_delta_theta = y_delta_theta

        slack_idx_val = int(slack_mask.nonzero(as_tuple=True)[0][0])
        bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
            edge_index_fwd, slack_idx_val, n_buses
        )
        data.bfs_edge_idx = bfs_edge_idx
        data.bfs_signs = bfs_signs
        data.bfs_node_order = bfs_node_order
        data.bfs_parent = bfs_parent

    return data


def scenario_to_data(
    network: pypsa.Network,
    open_lines: list[str] | None = None,
    rating_models: dict | None = None,
    use_pnom_share: bool = False,
    ptdf_branch_mode: str = "lines",
    angle_mode: str = "node",
) -> Data:
    """
    Build a PyG Data object for GNN inference WITHOUT a solved power flow.

    Unlike _create_graph_data, this function:
    - Does NOT set data.y (no ground truth targets)
    - Does NOT set data.y_line_p / y_line_q / y_ptdf (no PF results)
    - Uses only topology + static inputs (bus type flags, loads, v_set, p_nom)
    - Sets switch_active=0.0 for open_lines (contingency encoding)

    Used by grid_scenario.ScenarioBase.to_pyg() (Task B).
    rating_models: optional dict {line_name: StaticLineRating} for s_nom overrides.

    Source: NEW — not present in notebooks (simulation path only).
    """
    buses = list(network.buses.index)
    n_buses = len(buses)
    bus_to_i = {b: i for i, b in enumerate(buses)}

    # ── Bus type flags ────────────────────────────────────────────────────────
    is_slack = torch.zeros(n_buses, dtype=torch.float)
    is_pv = torch.zeros(n_buses, dtype=torch.float)
    is_pq = torch.zeros(n_buses, dtype=torch.float)
    slack_mask = torch.zeros(n_buses, dtype=torch.bool)
    pv_mask = torch.zeros(n_buses, dtype=torch.bool)
    pq_mask = torch.zeros(n_buses, dtype=torch.bool)

    for gen_name, gen in network.generators.iterrows():
        bus_name = gen["bus"]
        if bus_name not in bus_to_i:
            continue
        i = bus_to_i[bus_name]
        ctrl = gen["control"]
        if ctrl == "Slack":
            is_slack[i] = 1.0
            slack_mask[i] = True
        elif ctrl == "PV":
            is_pv[i] = 1.0
            pv_mask[i] = True

    for i in range(n_buses):
        if not slack_mask[i] and not pv_mask[i]:
            is_pq[i] = 1.0
            pq_mask[i] = True

    # ── Input features from loads + generator setpoints ───────────────────────
    # For pre-solve inference: p_set = nominal load injection (known before solve)
    p_inj = torch.zeros(n_buses, dtype=torch.float)
    q_inj = torch.zeros(n_buses, dtype=torch.float)
    for _, load in network.loads.iterrows():
        b = load["bus"]
        if b in bus_to_i:
            p_inj[bus_to_i[b]] -= float(load.get("p_set", 0.0))
            q_inj[bus_to_i[b]] -= float(load.get("q_set", 0.0))
    for _, gen in network.generators.iterrows():
        b = gen["bus"]
        if b in bus_to_i and gen["control"] in ("PV", "PQ"):
            p_inj[bus_to_i[b]] += float(gen.get("p_set", 0.0))

    # v_set from bus or generator v_mag_pu_set
    v_set = torch.ones(n_buses, dtype=torch.float)
    for gen_name, gen in network.generators.iterrows():
        b = gen["bus"]
        if b in bus_to_i:
            v_set[bus_to_i[b]] = float(gen.get("v_set", 1.0) or 1.0)

    # Slack angle reference = 0
    vang_ref = torch.zeros(n_buses, dtype=torch.float)

    # Mask unknowns (same convention as _create_graph_data)
    x_p = p_inj.clone()
    x_q = q_inj.clone()
    x_vmag = v_set.clone()
    x_vang = vang_ref.clone()
    x_p[slack_mask] = 0.0
    x_q[slack_mask] = 0.0
    x_q[pv_mask] = 0.0
    x_vmag[pq_mask] = 0.0
    x_vang[pq_mask] = 0.0
    x_vang[pv_mask] = 0.0

    x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)

    if use_pnom_share:
        p_nom_share_col = torch.zeros(n_buses, 1)
        p_noms = network.generators["p_nom"].values.astype(float)
        total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
        for gen_name, gen in network.generators.iterrows():
            b = gen["bus"]
            if b in bus_to_i:
                p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
        x = torch.cat([x, p_nom_share_col], dim=1)

    # ── Voltage limits ────────────────────────────────────────────────────────
    v_min_col = "v_mag_pu_min"
    v_max_col = "v_mag_pu_max"
    v_min_vals = (
        network.buses[v_min_col].values.astype(float)
        if v_min_col in network.buses.columns
        else np.full(n_buses, 0.9)
    )
    v_max_vals = (
        network.buses[v_max_col].values.astype(float)
        if v_max_col in network.buses.columns
        else np.full(n_buses, 1.1)
    )
    v_lim = torch.tensor(np.column_stack([v_min_vals, v_max_vals]), dtype=torch.float)

    # ── Edges ─────────────────────────────────────────────────────────────────
    (
        edge_index, edge_attr, fwd_mask, dc_fwd_mask,
        s_nom_edge, fwd_line_idx_list, ptdf_edge_row_idx_list,
    ) = _build_graph_edges(network, open_lines=open_lines, use_edge_features=True)

    # Override s_nom from rating_models if provided
    if rating_models is not None:
        branches = list(network.lines.index) + list(network.transformers.index)
        for edge_pos in range(0, edge_index.size(1), 2):  # forward edges only
            b0 = buses[edge_index[0, edge_pos].item()]
            b1 = buses[edge_index[1, edge_pos].item()]
            for br_name in rating_models:
                line = network.lines.get(br_name) if br_name in network.lines.index else None
                if line is not None and line["bus0"] == b0 and line["bus1"] == b1:
                    rating = rating_models[br_name].get_rating(network, br_name)
                    s_nom_edge[edge_pos, 0] = rating
                    s_nom_edge[edge_pos + 1, 0] = rating
                    break

    assert edge_attr.size(1) == EDGE_FEATURE_DIM

    line_idx = torch.tensor(fwd_line_idx_list, dtype=torch.long)
    ptdf_edge_row_idx = torch.tensor(ptdf_edge_row_idx_list, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.slack_mask = slack_mask
    data.pv_mask = pv_mask
    data.pq_mask = pq_mask
    data.forward_edge_mask = fwd_mask
    data.dc_flow_mask = dc_fwd_mask
    data.ptdf_line_index = line_idx
    data.ptdf_edge_row_idx = ptdf_edge_row_idx
    data.s_nom_edge = s_nom_edge
    data.v_lim = v_lim
    data.network_idx = torch.tensor([0], dtype=torch.long)
    # Note: data.y is NOT set — this is the PF-free inference path

    # ── BFS metadata for edge_delta angle mode ────────────────────────────────
    if angle_mode in ("edge_delta", "both"):
        all_fwd_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        all_fwd_mask[::2] = True
        edge_index_fwd = edge_index[:, all_fwd_mask]
        if slack_mask.any():
            slack_idx_val = int(slack_mask.nonzero(as_tuple=True)[0][0])
            bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
                edge_index_fwd, slack_idx_val, n_buses
            )
            data.bfs_edge_idx = bfs_edge_idx
            data.bfs_signs = bfs_signs
            data.bfs_node_order = bfs_node_order
            data.bfs_parent = bfs_parent

    return data


def collate_with_ptdf(batch: list[Data]) -> Batch:
    """
    Custom collate function for variable-size PTDF matrices and per-graph attributes.

    Source: GNN_Powerflow_V2.7_Training.ipynb cell 12
    TODO: remove duplicate from notebook once refactored.
    """
    y_ptdf_list = [data.y_ptdf for data in batch if hasattr(data, "y_ptdf")]
    y_line_p_list = [data.y_line_p for data in batch if hasattr(data, "y_line_p")]
    y_line_q_list = [data.y_line_q for data in batch if hasattr(data, "y_line_q")]
    ptdf_line_index_list = [
        data.ptdf_line_index for data in batch if hasattr(data, "ptdf_line_index")
    ]

    bfs_keys = ["bfs_edge_idx", "bfs_signs", "bfs_node_order", "bfs_parent", "y_delta_theta"]
    bfs_data: dict[str, list] = {k: [] for k in bfs_keys}
    for data in batch:
        for k in bfs_keys:
            if hasattr(data, k):
                bfs_data[k].append(getattr(data, k))
                delattr(data, k)

    for data in batch:
        for attr in ("y_ptdf", "y_line_p", "y_line_q", "ptdf_line_index"):
            if hasattr(data, attr):
                delattr(data, attr)

    batched = Batch.from_data_list(batch)

    if y_ptdf_list:
        batched.y_ptdf_list = y_ptdf_list
    if y_line_p_list:
        batched.y_line_p_list = y_line_p_list
    if y_line_q_list:
        batched.y_line_q_list = y_line_q_list
    if ptdf_line_index_list:
        batched.ptdf_line_index_list = ptdf_line_index_list

    for k, v_list in bfs_data.items():
        if v_list:
            setattr(batched, k, v_list)

    # Restore variable-size attrs on originals
    for i, data in enumerate(batch):
        if y_ptdf_list:
            data.y_ptdf = y_ptdf_list[i]
        if y_line_p_list:
            data.y_line_p = y_line_p_list[i]
        if y_line_q_list:
            data.y_line_q = y_line_q_list[i]
        if ptdf_line_index_list:
            data.ptdf_line_index = ptdf_line_index_list[i]
        for k in bfs_keys:
            if bfs_data[k]:
                setattr(data, k, bfs_data[k][i])

    for mask_attr in ("slack_mask", "pv_mask", "pq_mask"):
        if not hasattr(batched, mask_attr):
            raise RuntimeError(
                f"collate_with_ptdf: batched object missing '{mask_attr}'. "
                f"Ensure PowerFlowDataset stores data.{mask_attr} on every Data object."
            )

    return batched


def validate_training_data(
    networks: list,
    name: str = "dataset",
    n_sample: int = 2,
) -> bool:
    """
    Validate that a list of PyPSA networks is compatible with PowerFlowDataset.

    Source: GNN_Powerflow_V2.7_Training.ipynb cell 12
    TODO: remove duplicate from notebook once refactored.
    """
    issues: list[str] = []
    warnings: list[str] = []

    print(f"\n{'=' * 60}")
    print(f"Validating '{name}': {len(networks)} networks")
    print(f"{'=' * 60}")

    if len(networks) == 0:
        print("CRITICAL: Empty network list")
        return False

    for net_idx, net in enumerate(networks[:n_sample]):
        tag = f"[net {net_idx}]"
        print(f"\n--- Network {net_idx} ---")
        n_buses = len(net.buses)
        n_lines = len(net.lines)
        n_trafos = len(net.transformers)
        n_gens = len(net.generators)
        n_loads = len(net.loads)
        n_snaps = len(net.snapshots)
        print(
            f"  Buses={n_buses}, Lines={n_lines}, Trafos={n_trafos}, "
            f"Gens={n_gens}, Loads={n_loads}, Snapshots={n_snaps}"
        )

        if n_buses == 0:
            issues.append(f"{tag} No buses")
        if n_lines + n_trafos == 0:
            issues.append(f"{tag} No lines or transformers")
        if n_snaps == 0:
            issues.append(f"{tag} No snapshots")
        if n_gens == 0:
            warnings.append(f"{tag} No generators")

        for tbl, attr in [
            ("buses_t.p", "p"),
            ("buses_t.q", "q"),
            ("buses_t.v_mag_pu", "v_mag_pu"),
            ("buses_t.v_ang", "v_ang"),
        ]:
            df = getattr(net.buses_t, attr)
            ok = not df.empty and df.shape == (n_snaps, n_buses)
            finite = np.isfinite(df.values).all() if ok else False
            status = "✅" if (ok and finite) else "❌"
            if not ok:
                issues.append(f"{tag} {tbl} missing or wrong shape")
            elif not finite:
                issues.append(f"{tag} {tbl} contains NaN/Inf")
            print(f"  {status} {tbl}: shape={df.shape}")

        if "control" not in net.generators.columns:
            issues.append(f"{tag} net.generators missing 'control' column")
        else:
            ctrl_counts = net.generators["control"].value_counts().to_dict()
            has_slack = "Slack" in ctrl_counts
            status = "✅" if has_slack else "❌"
            print(f"  {status} control types: {ctrl_counts}")
            if not has_slack:
                issues.append(f"{tag} No Slack generator")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY for '{name}'")
    if issues:
        print(f"\n❌ {len(issues)} CRITICAL issue(s):")
        for iss in issues:
            print(f"   • {iss}")
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   • {w}")
    if not issues and not warnings:
        print("✅ All checks passed")
    elif not issues:
        print("✅ No critical issues")
    print(f"{'=' * 60}\n")
    return len(issues) == 0
