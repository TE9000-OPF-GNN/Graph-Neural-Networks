"""
Patch Step 2+3: Fix compute_ptdf_matrix (transformer B-matrix) + extend return + add DCBaseline classes.
Applied to GNN_Powerflow_V2.7_Training.ipynb cell 8.
"""
import json

NB_PATH = 'GNN_Powerflow_V2.7_Training.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[8]['source'])

# === PATCH 1: Fix compute_ptdf_matrix return and add transformer B-matrix ===
# Find the return statement in compute_ptdf_matrix
old_return = "    return PTDF"
new_return = """    return PTDF, B_red_inv, slack_idx, non_slack  # [STSI 230526]:extended return for DCBaselineComputer"""

assert src.count(old_return) == 1, f"Expected 1 occurrence of old return, found {src.count(old_return)}"
src = src.replace(old_return, new_return)

# Insert transformer B-matrix loop before "# Reduce B matrix (remove slack row/col)"
old_reduce = "    # Reduce B matrix (remove slack row/col)"
new_reduce = """    # [STSI 230526]:include transformer susceptance in B (tap_ratio=1.0 in flat-pu)
    for trafo in network.transformers.index:
        row = network.transformers.loc[trafo]
        x_val = float(row.get("x_pu_eff", row["x"])) if "x_pu_eff" in row.index and pd.notna(row.get("x_pu_eff")) else float(row["x"])
        if x_val == 0:
            continue
        b_val = 1.0 / x_val
        from_idx = bus_to_idx[row["bus0"]]
        to_idx = bus_to_idx[row["bus1"]]
        B[from_idx, from_idx] += b_val
        B[to_idx, to_idx] += b_val
        B[from_idx, to_idx] -= b_val
        B[to_idx, from_idx] -= b_val
        # NOT added to incidence A - PTDF gives line flows only

    # Reduce B matrix (remove slack row/col)"""

assert src.count(old_reduce) == 1, f"Expected 1 occurrence, found {src.count(old_reduce)}"
src = src.replace(old_reduce, new_reduce)

# === PATCH 2: Add DCBaselineStatic, DCBaselineComputer, precompute_baseline_statics ===
# Insert after compute_ptdf_matrix ends (after the blank lines following it)
# Find the marker between compute_ptdf_matrix and SECTION 3
old_section3 = """

# =============================================================================
# SECTION 3: ADMITTANCE MATRIX
# ============================================================================="""

new_section3 = """

# =============================================================================
# SECTION 2B: DC BASELINE INFRASTRUCTURE  [STSI 230526]:new section for PTDF-residual GNN
# =============================================================================

from dataclasses import dataclass

@dataclass
class DCBaselineStatic:
    \"\"\"Topology-static cache for one network. Computed once per topology via Phase 1.\"\"\"
    B_red_inv: np.ndarray        # [N-1, N-1]
    slack_idx: int
    non_slack_indices: list      # list[int], len = N-1
    base_q_gen: np.ndarray       # [N] Q_gen at base operating point (0 for non-gen buses)
    base_p_gen: np.ndarray       # [N] P_gen at base operating point (0 for non-gen buses)
    n_buses: int


class DCBaselineComputer:
    \"\"\"
    Swappable DC+Q baseline computer. Two-phase design:
      Phase 1 (build_topology_cache): O(N^2) matrix inversion, call once per topology.
      Phase 2 (compute_snapshot):     O(N) matmul + arithmetic, call per snapshot.
    Replace this class with a different baseline provider without touching GNN or training code.
    \"\"\"

    def build_topology_cache(self, network) -> DCBaselineStatic:
        \"\"\"Phase 1: build topology-static B_red_inv and base-case Q/P from network object.\"\"\"
        _, B_red_inv, slack_idx, non_slack_indices = compute_ptdf_matrix(network)
        n = len(network.buses)

        # Base-case P/Q per bus (from generator dispatch in base network)
        base_p_gen = np.zeros(n)
        base_q_gen = np.zeros(n)
        bus_order = list(network.buses.index)
        for gen_name, gen in network.generators.iterrows():
            bus_i = bus_order.index(gen.bus)
            # Use p_set as base dispatch (this is the nominal operating point)
            p_val = gen.p_set if hasattr(gen, 'p_set') and not np.isnan(gen.p_set) else 0.0
            base_p_gen[bus_i] += p_val
            q_val = gen.q_set if hasattr(gen, 'q_set') and not np.isnan(gen.q_set) else 0.0
            base_q_gen[bus_i] += q_val

        return DCBaselineStatic(
            B_red_inv=B_red_inv,
            slack_idx=slack_idx,
            non_slack_indices=non_slack_indices,
            base_q_gen=base_q_gen,
            base_p_gen=base_p_gen,
            n_buses=n,
        )

    def compute_snapshot(
        self,
        P_inj: np.ndarray,          # [N] signed net active injection (gen-load), pu
        Q_pq: np.ndarray,           # [N] known Q at PQ buses (0 elsewhere), pu
        Vmag_slack_pv: np.ndarray,  # [N] known Vmag for slack/PV (0 for PQ), pu
        slack_mask: np.ndarray,     # [N] bool
        pv_mask: np.ndarray,        # [N] bool
        pq_mask: np.ndarray,        # [N] bool
        static: DCBaselineStatic,
    ) -> tuple:
        \"\"\"
        Phase 2: pure numpy, no PyPSA. Returns (y_baseline [N,4], x_filled [N,7]).

        y_baseline layout: [Vmag_baseline, Vang_baseline, P_baseline, Q_baseline]
          - Known slots carry the ACTUAL known value (so y_true - y_baseline = 0 for knowns).
          - Unknown slots carry the DC/Q estimate.

        x_filled layout: [is_slack, is_PV, is_PQ, P, Q, Vmag, Vang] - same as V2.6 x,
          but unknown slots filled with DC baseline instead of 0.
        \"\"\"
        s = static
        N = s.n_buses

        # --- DC angles (Phase 2 core) ---
        P_nonslack = P_inj[s.non_slack_indices]
        theta = np.zeros(N)
        theta[s.non_slack_indices] = s.B_red_inv @ P_nonslack  # [N-1] radians

        # --- P_slack (lossless DC) ---
        P_slack_dc = -np.sum(P_inj[s.non_slack_indices])

        # --- Q crude (constant power factor) ---
        sum_Qd = np.sum(Q_pq[pq_mask])
        sum_Pd_raw = np.sum(-P_inj[pq_mask])  # P load (positive = consumed)
        sum_Pd = sum_Pd_raw if sum_Pd_raw > 1e-6 else 1.0

        # Generator delta from base
        gen_mask = ~pq_mask  # slack + PV
        delta_p_gen = P_inj - s.base_p_gen
        sum_delta_Pg = np.sum(delta_p_gen[gen_mask])
        sum_base_Pg = np.sum(s.base_p_gen[gen_mask])
        sum_base_Pg = sum_base_Pg if abs(sum_base_Pg) > 1e-6 else 1.0

        f = sum_delta_Pg / sum_base_Pg
        delta_Q_g_total = f * (sum_Qd / sum_Pd) * sum_delta_Pg if abs(sum_delta_Pg) > 1e-6 else 0.0

        # Allocate delta_Q across generator buses proportional to delta_P_g_i
        Q_gen_est = s.base_q_gen.copy()
        if abs(sum_delta_Pg) > 1e-6:
            Q_gen_est[gen_mask] += delta_Q_g_total * (delta_p_gen[gen_mask] / sum_delta_Pg)

        # --- Assemble y_baseline [N, 4]: [Vmag, Vang, P, Q] ---
        y_bl = np.zeros((N, 4), dtype=np.float32)

        # Vmag: PQ -> 1.0 (baseline); PV/slack -> Vmag_known (actual)
        y_bl[:, 0] = np.where(pq_mask, 1.0, Vmag_slack_pv)

        # Vang: slack -> 0 (actual reference); PV/PQ -> theta_dc (baseline)
        y_bl[:, 1] = theta
        y_bl[slack_mask, 1] = 0.0

        # P: slack -> P_slack_dc (baseline); PV/PQ -> P_known (actual)
        y_bl[:, 2] = P_inj
        y_bl[slack_mask, 2] = P_slack_dc

        # Q: PQ -> Q_known (actual); PV/slack -> Q_gen_est (baseline)
        y_bl[:, 3] = Q_pq
        y_bl[gen_mask, 3] = Q_gen_est[gen_mask]

        # --- Assemble x_filled [N, 7] ---
        x_filled = np.stack([
            slack_mask.astype(np.float32),
            pv_mask.astype(np.float32),
            pq_mask.astype(np.float32),
            y_bl[:, 2],   # col3 = P  (P_slack_dc for slack, P_known else)
            y_bl[:, 3],   # col4 = Q  (Q_gen_est for PV/slack, Q_known for PQ)
            y_bl[:, 0],   # col5 = Vmag (1.0 for PQ, Vmag_known else)
            y_bl[:, 1],   # col6 = Vang (theta_dc for PV/PQ, 0 for slack)
        ], axis=1).astype(np.float32)

        return y_bl, x_filled


def precompute_baseline_statics(networks, baseline_computer: DCBaselineComputer):
    \"\"\"
    Phase 1 for all networks. Mirrors precompute_Y_matrices pattern.
    Returns List[DCBaselineStatic], one per network (same index as Y_list).
    \"\"\"
    statics = []
    for net in networks:
        statics.append(baseline_computer.build_topology_cache(net))
    return statics


# =============================================================================
# SECTION 3: ADMITTANCE MATRIX
# ============================================================================="""

assert src.count(old_section3) == 1, f"Expected 1 occurrence of section3 marker, found {src.count(old_section3)}"
src = src.replace(old_section3, new_section3)

# Write back
cells[8]['source'] = [line + '\n' for line in src.split('\n')]
# Fix last line (no trailing newline)
if cells[8]['source']:
    cells[8]['source'][-1] = cells[8]['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("OK: Cell 8 patched (transformer B-matrix fix + DCBaseline classes)")
