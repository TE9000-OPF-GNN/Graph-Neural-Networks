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

## Constraints & Considerations

- **`y_ptdf` (PTDF matrix)** stays on Data — can still be used for line-flow auxiliary loss.
  The bilinear PTDF auxiliary head becomes redundant but is harmless at `weight_ptdf=0`.
- **Physics loss compatibility**: `p_inj[pq_mask] = x[pq_mask, 3]` reads known P from `x` (col 3).
  In new design col 3 for PQ is still `P_known`, so this line is **unchanged and safe**.
  Only need to reconstruct absolute Vmag/Vang/P_slack/Q_gen before mismatch computation.
- **`input_feature_filter`** rename to `fill_unknowns_with_baseline(node_features, bus_types, baseline)`.
  Same bus-type logic, but assigns baseline values instead of 0.
- **Base-case Q**: Requires that each PyPSA network has been solved at least once (true — DataGen runs AC PF for every snapshot). Use the **first snapshot** as base case per network.
- **Inference pipeline**: At inference time, compute DC baseline from known inputs, pass as `x`
  (unknowns filled with baseline), then: `y_abs = model(x) + y_baseline`. The baseline
  computation needs `B_red_inv` which must be stored alongside the network (or recomputed from PTDF).
- **Notebook series**: Create new `V2.7_DataGen`, `V2.7_Training`, `V2.7_Analysis` notebooks on
  the branch that copy/extend V2.6.x. Do NOT alter main-branch notebooks on this branch to
  keep the diff clean.

---

## Open Questions

- [ ] Q: Should `weight_ptdf` parameter be kept at 0 (backward compatible) or fully removed?
  Recommendation: keep at 0 to avoid merge conflicts, remove in a cleanup task later.
- [ ] Q: Should the angle reference constraint in physics loss (`pred[slack, 1].mean() = 0`)
  be relaxed since slack Vang=0 is now a KNOWN input (not a residual)? Answer: yes — slack Vang
  residual = 0 by construction, so no angle reference loss term needed. Simplifies physics loss.
- [ ] Q: For distributed slack (multiple slack buses), how to allocate `P_slack_DC`?
  Recommendation: allocate proportional to `p_nom` share (same as `pnom_share` feature).

---

## Recommendations

1. Create branch `feature/ptdf-residual-gnn` from current `main` HEAD (`91cb06f`).
2. New `compute_ptdf_and_dc_info(network)` in `code_base.py` returns `(ptdf_df, B_red_inv, slack_idx, non_slack_indices)`.
3. New `compute_dc_baseline(network, snapshot, ptdf_info, base_case_data=None)` returns `y_baseline [n_buses, 4]` and baseline-filled unknown slots.
4. Rename `input_feature_filter` → `fill_unknowns_with_baseline`; same signature but takes baseline array.
5. Add `y_baseline` to `Data` object in `_create_graph_data`; it's fixed-size so no collate changes.
6. In `physics_informed_loss_batch`: accept optional `use_residual_targets=False` flag; when True, look up `data.y_baseline` and reconstruct abs values.
7. New V2.7 notebooks on branch, copying V2.6.x structure. Training notebook: set `weight_ptdf=0`.
8. Inference helper `predict_from_dc_baseline(network, snapshot, model, ptdf_info)` to encapsulate the two-step: (1) compute baseline, (2) run GNN, (3) add baseline.
