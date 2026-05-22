# Research: PTDF-Residual GNN Feature (V2.7)

## Goal
Redesign the GNN so it predicts **residuals** between the true AC power flow solution and
a cheap DC+Q baseline (PTDF angles + constant-power-factor Q estimate), rather than
predicting absolute values. The DC baseline is passed as input for unknown variables
instead of zeros. A new git branch `feature/ptdf-residual-gnn` will isolate the work.

---

## Key Findings

### 1. Current `x` layout and masking — `code_base.py` line 1663 / 1745–1760
- **Location**: `PowerFlowDataset.input_feature_filter` (line 1663) and `_create_graph_data` (line ~1745)
- **What it does**: 7-column node features `[is_slack, is_PV, is_PQ, P, Q, Vmag, Vang]`.
  Unknown slots are **zeroed**: PQ→Vmag/Vang=0; PV→Q/Vang=0; Slack→P/Q=0.
- **Why it matters**: The zero-fill is what must be replaced with the DC/Q baseline.

### 2. Current `y` target — absolute AC values — line ~1776
- **Location**: `_create_graph_data`, `targets = np.column_stack([vmag, vang, p, q])`
- **What it does**: Stores absolute [Vmag, Vang, P, Q] from PyPSA solved PF.
- **Why it matters**: Must change to `y_true - y_baseline` (residuals only for unknown columns; 0 for known).

### 3. `y_ptdf` is the PTDF MATRIX, not DC angles — line 1800
- **Location**: `_create_graph_data`, `y_ptdf = torch.tensor(ptdf_df.values)  # [n_lines, n_buses]`
- **What it does**: Stores the full PTDF matrix as an auxiliary training target for the bilinear PTDF head.
- **Why it matters**: The PTDF auxiliary head (edge×node bilinear) becomes obsolete when PTDF becomes
  an input — `weight_ptdf` should be set to 0 on the new branch. `y_ptdf` attribute can be kept
  (still useful for line-flow auxiliary loss) but is NOT the DC angle solution.

### 4. DC angle computation needs `B_red_inv` — line 1522
- **Location**: `compute_ptdf_matrix`, line 1522 (`B_red_inv = np.linalg.inv(B_red_inv)`)
- **What it does**: Computes `B_red_inv` internally but only returns `PTDF = A_red @ B_red_inv`.
  Does **not** return `B_red_inv`.
- **Why it matters**: Need `theta_nonslack = B_red_inv @ P_nonslack` for the angle baseline.
  Fix: extend function to return `(ptdf_df, B_red_inv, slack_idx, non_slack)`, or add a new
  `compute_ptdf_and_dc_info(network)` companion.

### 5. `physics_informed_loss_batch` uses `pred` as absolute values — line 2041
- **Location**: `compute_power_flow_residual_from_pred` (line 2041) called via
  `physics_informed_loss_batch` (line 2145)
- **What it does**: Expects `pred[:, 0]=Vmag`, `pred[:, 1]=Vang`, `pred[:, 2]=P`, `pred[:, 3]=Q`
  as **absolute** quantities to compute Y×V mismatch.
- **Why it matters**: With residual targets, `pred` will be `Δ = y_true - y_baseline`.
  Physics loss must reconstruct: `pred_abs = pred + y_baseline` before computing mismatch.
  Need to pass `y_baseline` through the batch and into the residual function.
- **Also**: line `p_inj[pq_mask] = x[pq_mask, 3]` reads known P directly from `x`.
  In the new design `x[pq_mask, 3]` is still the known P (unchanged), so this is safe.

### 6. Existing `collate_with_ptdf` and variable-size handling — line 1846
- **Location**: `collate_with_ptdf` (line 1846)
- **What it does**: Strips `y_ptdf` from the Data object, batches, then re-attaches as a list.
  Also handles `y_line_p`. Fixed-size attributes go through PyG `Batch.from_data_list`.
- **Why it matters**: The new `y_baseline` tensor `[n_nodes, 4]` is **fixed-size-per-graph** and
  can be batched normally by PyG (no special handling needed — same shape as `y`).

### 7. PTDF bilinear head in `PowerFlowGNN` — V2.6.1 Training cell 11
- **Location**: `PowerFlowGNN.__init__` has `self.ptdf_W = nn.Parameter(...)`; `train_power_flow_gnn`
  has `_compute_ptdf_loss_matrix / _compute_ptdf_loss_flows`.
- **What it does**: Learns to predict the PTDF matrix as an auxiliary signal during training.
- **Why it matters**: This head has **no role** in the new design (PTDF is now input, not target).
  Set `weight_ptdf=0.0` (keeps code clean, costs nothing). Optionally remove the `ptdf_W` parameter
  entirely on the new branch.

---

## DC Baseline Formulas (mathematical specification)

Given: bus active power injections `P[i]` (from PyPSA before AC solve), slack bus index `s`.

**Angles**:
$$\theta_i = 0 \quad (i = s, \text{slack reference})$$
$$\boldsymbol{\theta}_{\text{non-slack}} = B_{\text{red}}^{-1} \cdot \mathbf{P}_{\text{non-slack}}$$

**P_slack** (lossless DC assumption):
$$P_{\text{slack,DC}} = -\sum_{i \neq s} P_i$$

**Vmag baseline** (no DC estimate):
$$V_{\text{mag},i} = 1.0 \text{ pu} \quad \forall i$$

**Q baseline** (constant power factor, eqs 15-18 from paper):

Let base-case values per bus be stored at dataset init time from the base network:
$$f = \frac{\sum_i \Delta P_{g,i}}{\sum_i P_{g,i}^{\text{base}}}
    = \frac{\sum_i P_{g,i}^{\text{scenario}} - \sum_i P_{g,i}^{\text{base}}}{\sum_i P_{g,i}^{\text{base}}}$$

Load Q (PQ buses — already KNOWN, so baseline = actual; residual = 0):
$$Q_{d,i}^{\text{est}} = Q_{d,i}^{\text{known}} \quad (i \in \text{PQ})$$

Generator Q increase (PV and slack buses):
$$\Delta Q_g = f \cdot \frac{\sum_{i \in \text{PQ}} Q_{d,i}}{\sum_{i \in \text{PQ}} P_{d,i}} \cdot \sum_i \Delta P_{g,i}$$

Allocate `ΔQ_g` across generators proportional to `ΔP_g_i`:
$$Q_{g,i}^{\text{est}} = Q_{g,i}^{\text{base}} + \Delta Q_g \cdot \frac{\Delta P_{g,i}}{\sum_j \Delta P_{g,j}}$$

Note: If `sum(ΔP_g) ≈ 0`, fall back to `Q_g_est = Q_g_base` (no change assumed).

---

## New `x` Input Layout (same 7 columns, new semantics for unknowns)

| Col | Name | Slack | PV | PQ |
|-----|------|-------|----|----|
| 0 | is_slack | 1 | 0 | 0 |
| 1 | is_PV | 0 | 1 | 0 |
| 2 | is_PQ | 0 | 0 | 1 |
| 3 | P | **P_slack_DC** (baseline) | P_known | P_known |
| 4 | Q | **Q_slack_crude** (baseline) | **Q_pv_crude** (baseline) | Q_known |
| 5 | Vmag | Vmag_known | Vmag_known | **1.0** (baseline) |
| 6 | Vang | 0 (reference, known) | **θ_dc** (baseline) | **θ_dc** (baseline) |

Bold = was 0, now DC/Q baseline.

## New `y` Target Layout (residuals for unknowns, 0 for knowns)

| Col | Name | Slack | PV | PQ |
|-----|------|-------|----|----|
| 0 | ΔVmag | Vmag_true − 1.0 | 0 | Vmag_true − 1.0 |
| 1 | ΔVang | 0 (reference) | θ_true − θ_dc | θ_true − θ_dc |
| 2 | ΔP | P_true − P_slack_DC | 0 | 0 |
| 3 | ΔQ | Q_true − Q_slack_crude | Q_true − Q_pv_crude | 0 |

Also store `data.y_baseline [n_buses, 4]` = [Vmag_baseline, θ_baseline, P_baseline, Q_baseline]
for reconstruction during inference: `y_pred_abs = y_pred_residual + y_baseline`.

---

## Patterns to Follow

| Pattern | Location | Notes |
|---------|----------|-------|
| PTDF computation with `B_red_inv` | `code_base.py:1482` | Extract `B_red_inv` from existing function |
| Collating fixed-size per-node tensors | `collate_with_ptdf` line 1846 | `y_baseline` batches normally through PyG |
| Physics loss reconstruction | `compute_power_flow_residual_from_pred` line 2041 | Add `y_baseline` param, compute `pred_abs = pred + baseline` before mismatch |
| Base-case storage pattern | `PowerFlowDataset.ptdf_matrices` line 1645 | Add `base_q_list`, `base_p_gen_list` per network |

---

## Key Files

| File | Purpose | Changes needed |
|------|---------|---------------|
| `code_base.py` | Core functions | (1) extend `compute_ptdf_matrix` → return `B_red_inv`; (2) new `compute_dc_baseline`; (3) `_create_graph_data`: fill unknowns with baseline, y=residuals, add `data.y_baseline`; (4) `physics_informed_loss_batch`: reconstruct absolute values |
| `GNN_Powerflow_V2.6_DataGen.ipynb` | Data generation | Minor: ensure base-case Q stored per network (already solved as first snapshot) |
| `GNN_Powerflow_V2.6.1_Training.ipynb` | Training + inference | Set `weight_ptdf=0`; update inference reconstruction; update loss to pass baseline |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | Analysis | Update prediction reconstruction calls |

---

## Critical Architectural Constraint: Inference-Time Modularity (added 2026-05-22)

**The core motivation is speed.** The DC baseline must be computable at inference time with NO
PyPSA dependency and must be fast enough that it does NOT negate the GNN's speed advantage.

### Two-phase separation (mandatory)

**Phase 1 — Topology-dependent, computed ONCE per network (not per snapshot):**
```
ptdf_info = compute_ptdf_and_dc_info(network_topology)
# Returns: (ptdf_df, B_red_inv [N-1, N-1], slack_idx, non_slack_indices)
# Cost: one (N-1)×(N-1) matrix inversion.
# ieee9: 8×8 ~microseconds; ieee118: 117×117 ~1–2 ms.
# Must be stored/serialized alongside the model for inference.
```

**Phase 2 — Snapshot-dependent, called per solve request (during inference):**
```
y_baseline, x_with_baseline = compute_dc_baseline(
    P_inj,          # [N] active power injections (known before AC solve)
    Q_pq,           # [N] reactive power for PQ buses (known)
    Vmag_known,     # [N] voltage setpoints (PV/slack, 0 for PQ)
    bus_masks,      # (slack_mask, pv_mask, pq_mask)
    ptdf_info,      # from Phase 1 (topology-static)
    base_case_q,    # [N] base-case Q per generator bus (from precomputation)
    base_case_p_gen # [N] base-case P per generator bus
)
# Returns: y_baseline [N, 4] and x_filled [N, 7]
# Cost: one (N-1)-dimensional matrix-vector multiply + scalar arithmetic.
# ieee9: ~microseconds; ieee118: ~microseconds.
# NO PYPSA CALL. Pure numpy/torch operations.
```

### Why this decomposition works
- `B_red_inv` depends only on network **topology** (line x values), not on operating point.
  Topology changes rarely (contingency analysis) or never (steady-state study).
- Per-snapshot cost is **O(N)** arithmetic + one O(N²) → O(N) matmul that is extremely cache-friendly.
- At training time: Phase 1 happens in `PowerFlowDataset.__init__`; Phase 2 in `_create_graph_data`.
- At inference time: Phase 1 runs once when the model is loaded; Phase 2 runs per prediction request.

### Modularity requirement
The `compute_dc_baseline` function must be a **pure function** (no side effects, no PyPSA calls)
with a well-defined interface so it can be swapped for an alternative baseline provider later
(e.g., fast linearized AC, warm-start from previous snapshot, learned baseline, etc.).

**Interface contract:**
```python
def compute_dc_baseline(
    P_inj: np.ndarray,       # [N] full bus active power injections (signed: gen>0, load<0)
    Q_pq: np.ndarray,        # [N] known reactive power at PQ buses (0 elsewhere)
    Vmag_slack_pv: np.ndarray, # [N] known Vmag for slack/PV buses (0 elsewhere)
    bus_masks: tuple,          # (slack_mask, pv_mask, pq_mask) bool arrays [N]
    B_red_inv: np.ndarray,   # [N-1, N-1] precomputed (topology-static)
    slack_idx: int,
    non_slack_indices: list,
    base_case_q_gen: np.ndarray,  # [N] Q_gen at base operating point (gen buses only)
    base_case_p_gen: np.ndarray,  # [N] P_gen at base operating point (gen buses only)
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (y_baseline [N,4], x_unknowns_filled [N,7])
    ...
```

### Timing benchmark plan
The implementation task must include a **standalone timing cell** in the training notebook:
```python
# Benchmark: baseline-only vs full AC PF
# Should show baseline << DC PyPSA << AC PyPSA
```
Expected hierarchy: baseline (~0.01–0.1 ms) < PyPSA DC lpf (~2.6 ms) < AC pf (~200 ms per snap).

---

## Constraints & Considerations

- **`y_ptdf` (PTDF matrix)** stays on Data — can still be used for line-flow auxiliary loss.
  The bilinear PTDF auxiliary head becomes redundant but is harmless at `weight_ptdf=0`.
- **Physics loss compatibility**: `p_inj[pq_mask] = x[pq_mask, 3]` reads known P from `x` (col 3).
  In new design col 3 for PQ is still `P_known`, so this line is **unchanged and safe**.
  Only need to reconstruct absolute Vmag/Vang/P_slack/Q_gen before mismatch computation.
- **`input_feature_filter`** rename to `fill_unknowns_with_baseline`; same signature but takes baseline array.
- **Base-case Q storage**: `PowerFlowDataset.__init__` stores `base_q_gen[net_idx]` and
  `base_p_gen[net_idx]` from the **first snapshot** of each network. These are the only two extra
  per-network arrays needed (both `[N]`-shaped, cheap to store).
- **Inference pipeline** (explicit sequence):
  1. Load model + `ptdf_info` (B_red_inv, slack_idx, non_slack_indices) — topology-static, load once.
  2. For each new operating point: call `compute_dc_baseline(P_inj, Q_pq, ...)` — pure numpy.
  3. Build `x` with baseline-filled unknowns. Build PyG `Data` object (no PTDF matrix needed).
  4. `pred_residual = model(data)` — GNN forward pass.
  5. `pred_abs = pred_residual + y_baseline` — scalar add.
- **`B_red_inv` serialization**: store as `.npy` file or JSON alongside saved model (in `run_info`).
- **Notebook series**: Create new `V2.7_DataGen`, `V2.7_Training`, `V2.7_Analysis` notebooks on
  the branch. Do NOT modify V2.6.x on this branch to keep the diff reviewable.

---

## Open Questions

- [ ] `weight_ptdf`: keep at 0 (backward compat) or remove? → Keep at 0 for now.
- [ ] Angle reference loss in physics (`pred[slack,1].mean()=0`): slack Vang residual is 0 by
  construction → term drops out naturally; no code change needed.
- [ ] Distributed slack P allocation: use `p_nom` share (same as `pnom_share` feature).
- [ ] **Timing**: need a benchmark cell to confirm baseline << DC PyPSA. Add to V2.7 training notebook.

---

## Recommendations

1. ✅ Branch `feature/ptdf-residual-gnn` created from `main` HEAD (`91cb06f`).
2. `compute_ptdf_and_dc_info(network)` → returns `(ptdf_df, B_red_inv, slack_idx, non_slack_indices)`.
   Refactor existing `compute_ptdf_matrix` to expose these intermediate results.
3. `compute_dc_baseline(P_inj, Q_pq, Vmag_known, bus_masks, B_red_inv, slack_idx, non_slack_indices, base_q_gen, base_p_gen)` — pure numpy, no PyPSA. Callable independently at inference time.
4. `fill_unknowns_with_baseline(node_features, bus_types, y_baseline)` replaces `input_feature_filter`.
5. Add `y_baseline [N,4]` to `Data` — batches normally through PyG (fixed size, no collate change).
6. `physics_informed_loss_batch`: add `y_baseline` reconstruction before mismatch.
7. New V2.7 notebooks. Training: `weight_ptdf=0`. Include baseline timing benchmark cell.
8. Inference wrapper `predict_with_dc_baseline(ptdf_info, base_case_data, P_inj, Q_pq, Vmag_known, bus_masks, model)` that encapsulates steps 2–5 above and is independently benchmarkable.
