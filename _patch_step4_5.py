"""
Patch Step 4+5: Update PowerFlowDataset __init__ and _create_graph_data for baseline.
Applied to GNN_Powerflow_V2.7_Training.ipynb cell 10.
"""
import json

NB_PATH = 'GNN_Powerflow_V2.7_Training.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[10]['source'])

# === PATCH 1: Update __init__ signature ===
old_init = """    def __init__(
        self,
        networks: list,
        use_edge_features: bool = True,
        use_pnet_balance: bool = True,   # [STSI100526] kept for backfill compat (ignored internally)
        use_pnom_share: bool = False,     # [STSI100526] col7: p_nom_share (was col7=p_bal + col8=pnom_share before 100526)
        transform=None,
        pre_transform=None,
        ptdf_branch_mode: str = "lines",  # "lines" or "all" — controls PTDF supervision scope
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.networks          = networks
        self.use_edge_features = use_edge_features
        self.use_pnom_share    = use_pnom_share
        self.ptdf_branch_mode  = ptdf_branch_mode #STSI260402: pass ptdf_branch_mode to _create_graph_data for edge feature control"""

new_init = """    def __init__(
        self,
        networks: list,
        use_edge_features: bool = True,
        use_pnet_balance: bool = True,   # [STSI100526] kept for backfill compat (ignored internally)
        use_pnom_share: bool = False,     # [STSI100526] col7: p_nom_share (was col7=p_bal + col8=pnom_share before 100526)
        transform=None,
        pre_transform=None,
        ptdf_branch_mode: str = "lines",  # "lines" or "all" — controls PTDF supervision scope
        baseline_computer=None,           # [STSI 230526]:DCBaselineComputer instance or None (V2.6 compat)
        baseline_statics=None,            # [STSI 230526]:List[DCBaselineStatic] from precompute_baseline_statics
    ):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.networks          = networks
        self.use_edge_features = use_edge_features
        self.use_pnom_share    = use_pnom_share
        self.ptdf_branch_mode  = ptdf_branch_mode #STSI260402: pass ptdf_branch_mode to _create_graph_data for edge feature control
        self._baseline_computer = baseline_computer  # [STSI 230526]:store for _create_graph_data
        self._baseline_statics  = baseline_statics   # [STSI 230526]:store for _create_graph_data"""

assert src.count(old_init) == 1, f"Expected 1 occurrence of old_init, found {src.count(old_init)}"
src = src.replace(old_init, new_init)

# === PATCH 2: Update compute_ptdf_matrix call site ===
old_ptdf_call = """        try:
            ptdf_matrix = compute_ptdf_matrix(network)
            y_ptdf = torch.tensor(ptdf_matrix, dtype=torch.float)
        except Exception:
            n_lines = len(network.lines)
            y_ptdf = torch.zeros((n_lines, n_buses), dtype=torch.float)"""

new_ptdf_call = """        try:
            ptdf_matrix, _, _, _ = compute_ptdf_matrix(network)  # [STSI 230526]:unpack 4-tuple (B_red_inv used via DCBaselineStatic)
            y_ptdf = torch.tensor(ptdf_matrix, dtype=torch.float)
        except Exception:
            n_lines = len(network.lines)
            y_ptdf = torch.zeros((n_lines, n_buses), dtype=torch.float)"""

assert src.count(old_ptdf_call) == 1, f"Expected 1 occurrence of ptdf call, found {src.count(old_ptdf_call)}"
src = src.replace(old_ptdf_call, new_ptdf_call)

# === PATCH 3: Replace the zero-fill block + x/y assembly with baseline-aware version ===
old_zero_fill = """        # Mask unknowns to zero in input (prevents trivial leakage)
        x_p[slack_mask]    = 0.0   # Slack P unknown (to be predicted)
        x_q[slack_mask]    = 0.0   # Slack Q unknown
        x_q[pv_mask]       = 0.0   # PV Q unknown
        x_vmag[pq_mask]    = 0.0   # PQ Vmag unknown
        x_vang[pq_mask]    = 0.0   # PQ Vang unknown
        x_vang[pv_mask]    = 0.0   # PV Vang unknown

        x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)"""

new_zero_fill = """        # [STSI 230526]:Compute DC baseline or fall back to V2.6 zero-fill
        if self._baseline_computer is not None and self._baseline_statics is not None:
            static = self._baseline_statics[net_idx]
            # Build numpy arrays for compute_snapshot (pre-masking values)
            P_inj_np  = x_p.numpy()       # signed net injection at this snapshot
            Q_pq_np   = x_q.numpy()       # known Q (non-zero only for PQ buses)
            Vmag_np   = x_vmag.numpy()    # known Vmag (PV/slack), still has PQ values here
            # Zero PQ Vmag for the baseline (PQ Vmag is unknown)
            Vmag_for_bl = Vmag_np.copy()
            Vmag_for_bl[pq_mask.numpy()] = 0.0
            y_bl_np, x_filled_np = self._baseline_computer.compute_snapshot(
                P_inj_np, Q_pq_np, Vmag_for_bl,
                slack_mask.numpy().astype(bool),
                pv_mask.numpy().astype(bool),
                pq_mask.numpy().astype(bool),
                static,
            )
            x_filled  = torch.tensor(x_filled_np, dtype=torch.float)
            y_baseline = torch.tensor(y_bl_np, dtype=torch.float)
            # x: use baseline-filled tensor (unknown slots get DC estimates instead of zeros)
            x = x_filled
        else:
            # V2.6 behaviour: zero out unknown slots
            x_p[slack_mask]    = 0.0   # Slack P unknown (to be predicted)
            x_q[slack_mask]    = 0.0   # Slack Q unknown
            x_q[pv_mask]       = 0.0   # PV Q unknown
            x_vmag[pq_mask]    = 0.0   # PQ Vmag unknown
            x_vang[pq_mask]    = 0.0   # PQ Vang unknown
            x_vang[pv_mask]    = 0.0   # PV Vang unknown
            x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)
            y_baseline = None"""

assert src.count(old_zero_fill) == 1, f"Expected 1 occurrence of zero-fill block, found {src.count(old_zero_fill)}"
src = src.replace(old_zero_fill, new_zero_fill)

# === PATCH 4: Replace y = torch.stack(...) and data = Data(...) ===
# Update target: compute residual y when baseline present
old_target = """        if self.use_pnom_share:  # [STSI100526] removed p_bal_col (redundant); GNN computes load sum internally"""

# We need to insert the y computation BEFORE the pnom_share block
# First, find what's between x assembly and pnom_share
old_y_and_pnom = """        if self.use_pnom_share:  # [STSI100526] removed p_bal_col (redundant); GNN computes load sum internally
            # col7: static p_nom share per gen bus (p_nom_i / Σp_nom)
            # Tells the model each generator's relative capacity for slack distribution.
            p_nom_share_col = torch.zeros(n_buses, 1)
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
            x = torch.cat([x, p_nom_share_col], dim=1)

        # ── Target: full post-solve state for all buses ──────────────────────────
        y = torch.stack([v_mag, v_ang, p_bus, q_bus], dim=1)"""

new_y_and_pnom = """        if self.use_pnom_share:  # [STSI100526] removed p_bal_col (redundant); GNN computes load sum internally
            # col7: static p_nom share per gen bus (p_nom_i / Σp_nom)
            # Tells the model each generator's relative capacity for slack distribution.
            p_nom_share_col = torch.zeros(n_buses, 1)
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
            x = torch.cat([x, p_nom_share_col], dim=1)

        # ── Target: residual y (y_true - y_baseline) or absolute y ───────────  [STSI 230526]
        y_true = torch.stack([v_mag, v_ang, p_bus, q_bus], dim=1)
        if y_baseline is not None:
            y = y_true - y_baseline  # residual targets; known slots -> ~0 by construction
        else:
            y = y_true"""

assert src.count(old_y_and_pnom) == 1, f"Expected 1 occurrence of y_and_pnom, found {src.count(old_y_and_pnom)}"
src = src.replace(old_y_and_pnom, new_y_and_pnom)

# === PATCH 5: Add y_baseline to Data object after assembly ===
old_data_network_idx = """        # Graph-level metadata
        data.network_idx    = torch.tensor([net_idx], dtype=torch.long)

        return data"""

new_data_network_idx = """        # [STSI 230526]:store y_baseline for residual->absolute reconstruction in physics loss + eval
        if y_baseline is not None:
            data.y_baseline = y_baseline   # [N, 4], auto-batched by PyG Batch.from_data_list

        # Graph-level metadata
        data.network_idx    = torch.tensor([net_idx], dtype=torch.long)

        return data"""

assert src.count(old_data_network_idx) == 1, f"Expected 1 occurrence of data.network_idx block, found {src.count(old_data_network_idx)}"
src = src.replace(old_data_network_idx, new_data_network_idx)

# Write back
cells[10]['source'] = [line + '\n' for line in src.split('\n')]
if cells[10]['source']:
    cells[10]['source'][-1] = cells[10]['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("OK: Cell 10 patched (PowerFlowDataset baseline support + residual y)")
