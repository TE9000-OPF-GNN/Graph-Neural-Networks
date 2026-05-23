# Research: PTDF-Residual GNN Feature (V2.7)
## Source notebooks: V2.6.1_Training + V2.6_DataGen + V2.6_Analysis (verified 2026-05-23)
## NOTE: code_base.py is an old backup — do NOT use for implementation reference.

## Goal
Redesign the GNN so it predicts **residuals** between the true AC power flow solution and
a cheap DC+Q baseline (PTDF angles + constant-power-factor Q estimate), rather than
predicting absolute values. The DC baseline is passed as input for unknown variables
instead of zeros. Branch `feature/ptdf-residual-gnn` isolates the work.

---

## Key Findings (all verified against actual notebook cells)

### 1. `x` layout and masking — Training cell 10, `_create_graph_data`
- **What it does**: 7-column node features `[is_slack(0), is_PV(1), is_PQ(2), P(3), Q(4), Vmag(5), Vang(6)]`.
  Unknown slots are **zeroed inline** (no separate filter function — it's just direct assignment):
  ```python
  x_p[slack_mask] = 0.0;  x_q[slack_mask] = 0.0   # Slack P/Q unknown
  x_q[pv_mask]    = 0.0;  x_vang[pv_mask] = 0.0   # PV Q/Vang unknown
  x_vmag[pq_mask] = 0.0;  x_vang[pq_mask] = 0.0   # PQ Vmag/Vang unknown
  x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)
  ```
  Optional col 7: `p_nom_share` appended when `use_pnom_share=True`.
- **Why it matters**: These six `= 0.0` lines are exactly what gets replaced with DC baseline values.
  No separate `input_feature_filter` function exists in current notebooks.

### 2. `y` target — absolute AC values — Training cell 10
- **Exact line**: `y = torch.stack([v_mag, v_ang, p_bus, q_bus], dim=1)`
- **What it does**: Stores absolute [Vmag, Vang, P, Q] from PyPSA solved PF.
- **Why it matters**: Must change to `y_true - y_baseline`. Because `y_baseline` carries the **actual
  known value** for known slots (see Finding 2b), `y_true - y_baseline = 0` for known slots naturally.
  The residual target is therefore: non-zero only where the variable is unknown.

### 2b. `y_baseline [N, 4]` carries KNOWN values for known slots — CRITICAL
- **What it is**: `y_baseline[i, col]` = DC/Q estimate for unknown variables; **actual known value**
  for known variables. This dual meaning is intentional and required.
- **Full layout**:
  | Col | PQ | PV | Slack |
  |-----|----|----|-------|
  | 0 Vmag | 1.0 (baseline) | Vmag_known (actual) | Vmag_known (actual) |
  | 1 Vang | θ_dc (baseline) | θ_dc (baseline) | 0 (actual, reference) |
  | 2 P | P_known (actual) | P_known (actual) | P_slack_DC (baseline) |
  | 3 Q | Q_known (actual) | Q_pv_crude (baseline) | Q_slack_crude (baseline) |
- **Why it matters**: Reconstruction `pred_abs = node_pred + y_baseline` works for **both** head modes:
  - Unknown slot: `pred_residual + baseline_estimate = pred_abs` ✓
  - Known slot: `0 + known_value = known_value` ✓ (critical for `with_encoder` — see Finding 9)
  - `y_true - y_baseline = 0` for known slots ✓ (clean residual target)

### 3. `y_ptdf` is the PTDF MATRIX — Training cell 10
- **Exact lines**:
  ```python
  ptdf_matrix = compute_ptdf_matrix(network)      # numpy [n_lines, n_buses]
  y_ptdf = torch.tensor(ptdf_matrix, dtype=torch.float)
  ```
- **What it does**: Stores the full DC PTDF matrix as auxiliary training target.
- **Why it matters**: `y_ptdf` is NOT DC angle predictions — it is the sensitivity matrix.
  On the new branch `physics_cfg.use_ptdf_loss=False` / `weight_ptdf=0` disables the auxiliary
  head. `y_ptdf` stays on the Data object (harmless, may be useful for line-flow loss).

### 4. `compute_ptdf_matrix` discards `B_red_inv` — Training cell 8
- **Location**: Training cell 8, function `compute_ptdf_matrix(network)`
- **Returns**: plain numpy array `PTDF [n_lines, n_buses]`. NOT a DataFrame.
  `B_red_inv`, `slack_idx`, `non_slack` are computed but **thrown away**.
- **Why it matters**: DC angle baseline needs `theta_nonslack = B_red_inv @ P_nonslack`.
  The fix is to extend return to `(PTDF, B_red_inv, slack_idx, non_slack_indices)`.
- **Important**: Called lazily inside `_create_graph_data` (one call per graph item).
  `PowerFlowDataset.__init__` does NOT precompute/cache PTDF — no `self.ptdf_matrices`.

### 4b. CRITICAL BUG: `compute_ptdf_matrix` omits transformers from B matrix (prerequisite fix)
- **Bug**: The function only loops over `network.lines` for susceptance. Transformers are
  completely omitted from B. For ieee9, gen buses (1,2,3) are ONLY connected via transformers
  → B has zero rows for those buses → `B_red` is singular → `pinv` → garbage angles.
- **Affected networks**: ieee9 (3 trafos), cigre14 (2), ieee30 (4), ieee39, ieee57, ieee118.
- **Fix (prerequisite for V2.7)**: After the lines loop, add transformer susceptance to B:
  ```python
  for trafo in network.transformers.index:
      row = network.transformers.loc[trafo]
      x_val = float(row.get("x_pu_eff", row["x"]))
      if x_val == 0: continue
      b_val = 1.0 / x_val
      from_idx = bus_to_idx[row["bus0"]]
      to_idx = bus_to_idx[row["bus1"]]
      B[from_idx, from_idx] += b_val
      B[to_idx, to_idx] += b_val
      B[from_idx, to_idx] -= b_val
      B[to_idx, from_idx] -= b_val
      # NOT added to incidence A — PTDF gives line flows only
  ```
- **Validation**: After fix, DC angles must match PyPSA `lpf()` angles.
- **Research reference**: `.docs/research/2026-05-23-compute-ptdf-matrix-missing-transformers.md` (main branch)

### 5. `compute_power_flow_residual_from_pred` uses `pred` as absolute values — Training cell 11
- **Signature** (current):
  ```python
  def compute_power_flow_residual_from_pred(pred, x, Y_matrix, network,
      bus_masks=None, use_q_partial_mode=False, w_P=0.5, w_Q=0.5)
  ```
- **Returns**: 3-tuple `(physics_loss, p_res_mean, q_res_mean)` — NOT a single scalar.
- **What it does**: Assembles absolute voltages/injections from `pred` and `x`:
  ```python
  v_mag[pq_mask]    = pred[pq_mask, 0]   # predicted
  v_ang[pv_mask]    = pred[pv_mask, 1]   # predicted
  p_inj[pq_mask]    = x[pq_mask, 3]     # KNOWN — reads from x col 3
  q_inj[pq_mask]    = x[pq_mask, 4]     # KNOWN — reads from x col 4
  ```
- **Why it matters**: With residual targets `pred` = `Δ`, the physics loss needs
  `pred_abs = pred + y_baseline` before computing Y×V mismatch.
  The `x[pq_mask, 3/4]` reads for known P/Q are **safe and unchanged** in new design
  (col 3 for PQ is still P_known).

### 6. `collate_with_ptdf` — Training cell 10
- **What it strips** before `Batch.from_data_list`: `y_ptdf`, `y_line_p`, `ptdf_line_index`
  (all variable-size). Restores as `batched.y_ptdf_list`, etc.
- **What it auto-batches** (fixed node-size): `x`, `y`, `slack_mask`, `pv_mask`, `pq_mask`,
  `edge_index`, `edge_attr`, `ptdf_edge_row_idx`, `network_idx`.
- **Why it matters**: The new `y_baseline [n_buses, 4]` is **fixed node-size** — it will be
  auto-batched by `Batch.from_data_list` with NO changes to `collate_with_ptdf`.

### 7. `PhysicsConfig` controls PTDF aux loss — Training cell 11
- **Current defaults**: `use_ptdf_loss=False`, `weight_ptdf=0.0`
- **What it does**: Gates the PTDF auxiliary head in training. Already off by default.
- **Why it matters**: New branch simply leaves `use_ptdf_loss=False`. No code removal needed.

### 8. `_masked_mse_loss` restricts MSE to unknowns — Training cell 14
- The training loop uses `_masked_mse_loss(node_pred, batch.y, batch)`, NOT raw `F.mse_loss`.
  It only penalises unknown variables per bus type (same mask logic as physics loss).
- **Why it matters**: With residual targets `y = y_true - y_baseline`, the masked MSE
  will correctly compare `pred_residual` to `y_residual` — **no change needed here**.
  The mask still applies correctly because known variables have residual = 0 in y.

### 9. `evaluate_dc_baseline` in Analysis cell 14 — COMPARATOR ONLY
- Calls PyPSA `lpf()`, collects timings, returns accuracy dict.
- **NOT** used as input to GNN. This is a reference timing/accuracy baseline for plots.
- **Why it matters**: This function stays unchanged. Inference uses a separate path.

### 11. DC baseline and PTDF physics loss are independently gated — verified Training cells 8/12/14

**Two entirely separate paths sharing only `compute_ptdf_matrix` as a computation source:**

| | DC Baseline (new, V2.7) | PTDF Physics Loss (existing, V2.6) |
|-|------------------------|-----------------------------------|
| **Source fn** | `compute_ptdf_matrix` → uses `B_red_inv` | `compute_ptdf_matrix` → uses `PTDF matrix` |
| **Stored on Data** | `data.y_baseline [N,4]` | `data.y_ptdf [n_lines, n_buses]` |
| **Consumed by** | `_create_graph_data` fill + physics loss reconstruction | `compute_ptdf_loss_matrix` / `compute_ptdf_loss_flows` |
| **Toggle** | Always active when `baseline_computer` provided | `physics_cfg.use_ptdf_loss` (default **False**) |
| **Reads from batch** | `batch.y_baseline` | `batch.y_ptdf_list`, `batch.y_line_p_list` |
| **Never reads** | `batch.y_ptdf_list` | `batch.y_baseline`, `B_red_inv` |

**Verified**: `compute_ptdf_loss_matrix` reads only `batch.y_ptdf_list[g]` + GNN embeddings.
`compute_ptdf_loss_flows` reads only `batch.y_line_p_list`, `batch.ptdf_line_index_list`,
and `batch.x[node_mask, 3]` (P injections). Neither touches `B_red_inv` or `y_baseline`.

**One subtle cross-point (not a coupling — just a V2.7 semantic change):**
`compute_ptdf_loss_flows` builds `delta_p_g = x_g[:, 3]`. In V2.6, `x[slack, 3] = 0`
(unknown P zeroed). In V2.7, `x[slack, 3] = P_slack_DC` (DC balance estimate).
The flow-space PTDF loss therefore becomes slightly more physically correct in V2.7 even
without any code change — this is a **beneficial side effect**, not a coupling issue.

**Redundancy hypothesis (not yet tested):**
- `compute_ptdf_loss_matrix`: supervises the bilinear `ptdf_W` head to match the DC sensitivity
  matrix. This is topology-based and independent of the operating point. May retain value even
  with DC baseline input.
- `compute_ptdf_loss_flows`: computes `ptdf_bilinear @ P_injections` vs `true_AC_line_flows`.
  If the model already receives `θ_dc = B_red_inv @ P_nonslack` as input (x[:,6] for non-slack),
  the DC flow information is already encoded in x. The flow-space loss is then teaching
  the bilinear head to reproduce something the input already implicitly contains — potentially
  **redundant**. Hypothesis: flow-space PTDF loss adds less benefit in V2.7 than in V2.6.
- **Safe to test**: set `use_ptdf_loss=False` (already the default) → baseline still computed,
  all baseline paths unaffected. The PTDF loss can be re-enabled independently at any time.

### 10. `head_mode="with_encoder"` is the preferred head — Training cell 11, `PowerFlowGNN`
- **What it does**: `node_pred = torch.zeros(N, 4, ...)`, then fills ONLY unknown slots per bus type:
  ```python
  node_pred[pq_mask,  0:2] = pq_head(h[pq_mask])    # col0=Vmag, col1=Vang
  node_pred[pv_mask,  1]   = pv_head(h[pv_mask])[:,0] # col1=Vang
  node_pred[pv_mask,  3]   = pv_head(h[pv_mask])[:,1] # col3=Q
  node_pred[slack_mask,2]  = slack_head(h[slack_mask])[:,0] # col2=P
  node_pred[slack_mask,3]  = slack_head(h[slack_mask])[:,1] # col3=Q
  ```
  Output is **structurally zero for known slots** — no prediction head for those.
  Output shape is still `[N, 4]` — same as `standard`.
- **Why it matters for residual design**:
  1. `with_encoder` output = `[pred_residual_for_unknowns, 0_for_knowns]`. Adding `y_baseline`
     (which has `known_value` in known slots) reconstructs the full absolute state naturally.
  2. `_masked_mse_loss` becomes **redundant** for `with_encoder` — the model structurally cannot
     predict known variables, so MSE is naturally zero there without masking. The mask still
     does no harm (it skips zero-residual entries), but could be replaced with plain `F.mse_loss`
     for `with_encoder` without loss of correctness.
  3. The physics loss reads known P/Q/Vmag/Vang directly from `x` (not from `pred`) — this is
     unchanged and correct for both head modes.
  4. `evaluate_gnn_on_test_set` (cells 17/Analysis 14) currently overrides known variables after
     inference: `vmag_eval[pv] = x[pv, 5]` etc. With `with_encoder`, these overrides are still
     correct (model outputs 0 there, baseline adds known value — but the explicit override in the
     eval function ensures ground-truth inputs regardless of reconstruction path).
- **Implication for `compute_snapshot` output** (`x_filled`): the `x_filled` returned by
  `DCBaselineComputer.compute_snapshot` replaces unknown-slot zeros with baseline estimates.
  For `with_encoder`, the model's node embedding still processes all 7 x-columns — the baseline
  values in x-unknowns replace zeros as richer starting signal, improving gradient flow.

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
| 0 | ΔVmag | 0 (known) | 0 (known) | Vmag_true − 1.0 |
| 1 | ΔVang | 0 (known reference) | θ_true − θ_dc | θ_true − θ_dc |
| 2 | ΔP | P_true − P_slack_DC | 0 (known) | 0 (known) |
| 3 | ΔQ | Q_true − Q_slack_crude | Q_true − Q_pv_crude | 0 (known) |

Also store `data.y_baseline [n_buses, 4]` = the baseline used above (see Finding 2b for exact values).
Reconstruction: `pred_abs = node_pred + y_baseline` — works identically for both head modes.

## Head-mode interaction with residual design

| Head mode | Known slots in output | `_masked_mse_loss` | Reconstruction |
|-----------|----------------------|--------------------|----------------|
| `standard` | Predicts non-zero (noise) | **Required** to block loss on known slots | `node_pred + y_baseline`; known slots: `noise + known ≈ known` (imprecise but MSE not trained on them) |
| `with_encoder` (preferred) | Structurally 0 by construction | Redundant but harmless | `node_pred + y_baseline`; known slots: `0 + known = known` exactly ✓ |

**Preferred configuration**: `head_mode="with_encoder"` + residual targets. The known-slot
reconstruction is exact, `_masked_mse_loss` can be simplified or kept for backward compat.

---

## Patterns to Follow (notebook-accurate)

| Pattern | Notebook | Cell | Notes |
|---------|----------|------|-------|
| Inline zero-masking of unknowns | V2.6.1_Training | cell 10 | 6 `x_?[mask] = 0.0` lines → replace with DC baseline |
| PTDF computation | V2.6.1_Training | cell 8 | Extend `compute_ptdf_matrix` to also return `B_red_inv, slack_idx, non_slack` |
| `y` target stack | V2.6.1_Training | cell 10 | `torch.stack([v_mag, v_ang, p_bus, q_bus])` → subtract `y_baseline`; result = 0 for known slots |
| `y_baseline` known slots | V2.6.1_Training | cell 10 | Known slots must hold the ACTUAL known value (see Finding 2b), not just DC estimates |
| `collate_with_ptdf` | V2.6.1_Training | cell 10 | `y_baseline` [N,4] auto-batched — no change needed |
| Physics loss 3-tuple | V2.6.1_Training | cell 11 | Add `y_baseline` param, reconstruct abs before Y×V |
| `with_encoder` forward | V2.6.1_Training | cell 11 | Known slots are structurally 0 in output; `pred_abs = pred + y_baseline` gives exact known values |
| PTDF aux loss gate | V2.6.1_Training | cell 11/14 | `use_ptdf_loss=False` already default — no change |
| DC baseline comparator | V2.6_Analysis | cell 14 | `evaluate_dc_baseline` calls `lpf()` — leave unchanged, it's a comparator |

---

## Key Files (notebook-accurate)

| File | Scope | Changes needed |
|------|-------|---------------|
| `GNN_Powerflow_V2.6.1_Training.ipynb` | **Primary** | cell 8: extend `compute_ptdf_matrix`; cell 10: `_create_graph_data` (6 masked zero lines + y + Data); cell 11: `compute_power_flow_residual_from_pred` add baseline param; cell 15: inference reconstruction |
| `GNN_Powerflow_V2.6_DataGen.ipynb` | **Minor** | Ensure base-case Q captured per network if DataGen needs to regenerate data |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | **Minor** | Inference calls: `pred_abs = pred + y_baseline`; `evaluate_dc_baseline` stays unchanged |

**Note**: `code_base.py` is an old backup — changes go into the notebook cells only.

---

## Critical Architectural Constraint: BaselineComputer Protocol (added 2026-05-23)

**Three priority rules (binding, in order):**
1. **P1 — Inference visibility**: baseline computation must be an explicit, timed step in
   inference — NOT reuse of precomputed stored data. The whole point is to measure real cost.
2. **P2 — Swappability**: `BaselineComputer` must be a well-defined swappable class so it can
   be replaced by a cheaper or more accurate method without touching GNN or training code.
3. **P3 — Fast training if possible** — allowed ONLY when it doesn't violate P1/P2.
   Allowed: precompute `y_baseline` during dataset build, store on `Data` object.
   The same `BaselineComputer.compute_snapshot(...)` call path is used at both build and inference time.

### `BaselineComputer` class design

```python
@dataclass
class DCBaselineStatic:
    """Topology-static cache for one network. Computed once per topology."""
    B_red_inv: np.ndarray        # [N-1, N-1]
    slack_idx: int
    non_slack_indices: list[int]
    base_q_gen: np.ndarray       # [N] Q_gen at nominal operating point
    base_p_gen: np.ndarray       # [N] P_gen at nominal operating point
    n_buses: int

class DCBaselineComputer:
    """
    Swappable baseline computer.  Implements the BaselineComputer protocol.
    Two-phase: build topology cache once, then call compute_snapshot per solve.
    Can be replaced by faster/better implementation without touching GNN code.
    """
    def build_topology_cache(self, network) -> DCBaselineStatic:
        """Phase 1: topology-static, O(N^2) inversion. Cheap per topology."""
        ...
    def compute_snapshot(
        self,
        P_inj, Q_pq, Vmag_slack_pv, bus_masks, static: DCBaselineStatic
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase 2: per-snapshot, O(N) matmul. Returns (y_baseline [N,4], x_filled [N,7])."""
        ...
```

### Training path (P3 optimization — precompute once, not per batch):
1. `PowerFlowDataset.__init__` calls `baseline_computer.build_topology_cache(net)` for each
   unique network topology. Stores `List[DCBaselineStatic]`, indexed same as `Y_list`.
   Template: mirrors the `precompute_Y_matrices` pattern in Training cell 8.
2. `_create_graph_data` calls `baseline_computer.compute_snapshot(...)` using the cached
   static for that network's `net_idx`. This is the **same code path as inference**.
3. Result `y_baseline [N,4]` stored as `data.y_baseline` on each Data object (auto-batched).

### Inference path (P1 — explicit timed step):
```python
# In evaluate_gnn_on_test_set, inside the per-snapshot loop:
t_baseline_start = time.perf_counter()
y_baseline_np, x_filled = baseline_computer.compute_snapshot(P_inj, Q_pq, ..., static[net_idx])
t_baseline = time.perf_counter() - t_baseline_start   # measured separately

t_solve_start = time.perf_counter()
node_pred = model(data)                                # residual prediction
t_solve = time.perf_counter() - t_solve_start

pred_abs = node_pred + torch.from_numpy(y_baseline_np)  # reconstruction
# t_total = t_baseline + t_solve + t_postproc
```
The `baseline_time_ms` must appear as a separate entry in `metrics` alongside `solve_time_ms`.

### `precompute_Y_matrices` is the template
`precompute_Y_matrices(networks, device)` in Training cell 8 returns `List[(Y_real, Y_imag)]`,
one tuple per network. The baseline static cache follows the exact same pattern:
`precompute_baseline_statics(networks, baseline_computer)` → `List[DCBaselineStatic]`.
This list is passed into `PowerFlowDataset` and also stored on the model's export dict.

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

## Recommendations (notebook-accurate, revised 2026-05-23 with BaselineComputer protocol)

1. ✅ Branch `feature/ptdf-residual-gnn` created from `main` HEAD.

2. **Training cell 8** — new `DCBaselineStatic` dataclass + `DCBaselineComputer` class:
   - `build_topology_cache(network)`: calls updated `compute_ptdf_matrix`, stores `B_red_inv`,
     `slack_idx`, `non_slack_indices`, `base_q_gen`, `base_p_gen` from first snapshot.
   - `compute_snapshot(P_inj, Q_pq, Vmag_slack_pv, bus_masks, static)`: pure numpy, no PyPSA.
   - `precompute_baseline_statics(networks, baseline_computer)` → `List[DCBaselineStatic]`.
     Mirrors `precompute_Y_matrices` pattern exactly.
   - Also extend `compute_ptdf_matrix` to return `(PTDF, B_red_inv, slack_idx, non_slack)`.

3. **Training cell 10** — `PowerFlowDataset.__init__`:
   - Accept `baseline_computer: DCBaselineComputer | None = None` param.
   - If provided: precompute `self._baseline_statics = precompute_baseline_statics(networks, bc)`.
   - `_create_graph_data`: call `bc.compute_snapshot(...)` using `self._baseline_statics[net_idx]`.
   - Replace 6 `x_?[mask] = 0.0` lines with DC baseline values (same positional columns).
   - Change `y = torch.stack([v_mag, v_ang, p_bus, q_bus])` → residual targets (unknowns only).
   - Add `data.y_baseline = torch.tensor(y_baseline, dtype=torch.float)` (auto-batched by PyG).

4. **Training cell 11** — `physics_informed_loss_batch` (NOT `compute_power_flow_residual_from_pred`):
   - Add `y_baseline: Optional[torch.Tensor] = None` param to `physics_informed_loss_batch`.
   - Inside the per-graph loop, AFTER `pred_g = pred[node_mask]` (line 430), ADD:
     ```python
     if y_baseline is not None:
         pred_g = pred_g + y_baseline[node_mask]  # residual → absolute
     ```
   - This converts residuals to absolute BEFORE the mixed-state assembly (lines 444-461).
   - `compute_power_flow_residual_from_pred` stays UNCHANGED — it receives `pred_g_mixed`
     which is already absolute after the assembly.
   - **Why not in `compute_power_flow_residual_from_pred`?** That function receives the
     already-assembled `pred_g_mixed` (known slots overridden from x). Adding y_baseline
     there would double-add known values.

5. **Training cell 14** — `train_power_flow_gnn`:
   - Add `baseline_computer: DCBaselineComputer | None = None` param.
   - Pass through to `PowerFlowDataset` for all splits (train/val/test).
   - Training loop: pass `y_baseline=batch.y_baseline` to `physics_informed_loss_batch`.
   - **Validation loop** (lines 491-496): same change — pass `y_baseline=batch.y_baseline`
     to `physics_informed_loss_batch` call in the validation section.

6. **Training cell 17** — `evaluate_gnn_on_test_set` (inference timing):
   - Add `baseline_computer` param and `static_cache: List[DCBaselineStatic]` param.
   - Inside per-snapshot loop: time `baseline_computer.compute_snapshot(...)` separately.
   - Report `baseline_time_ms` in returned `metrics` dict.
   - `pred_abs = node_pred + torch.from_numpy(y_baseline).to(device)` before evaluation.
   - **Do NOT** read `data.y_baseline` from stored dataset — recompute fresh from `compute_snapshot`.
   - **Ground truth reconstruction**: Since `data.y` now stores RESIDUALS, the eval function
     must reconstruct absolute truth: `y_true_abs = data.y + data.y_baseline` at lines 116-119
     (where `vmag_true = data.y[:, 0]` etc.). Without this, MAE/RMSE would compare absolute
     predictions against residual targets — completely wrong.

7. **Analysis cell 14** — `evaluate_dc_baseline` stays unchanged (comparator for DC PyPSA).
   The Analysis copy of `evaluate_gnn_on_test_set` needs same changes as Training cell 17.

8. **Timing benchmark cell** — add to V2.7 training notebook to confirm baseline << DC lpf << AC pf.

---

## Open Questions (updated 2026-05-23)
- [ ] Q baseline formula: confirm `base_q_gen` = first-snapshot Q for each generator, or nominal
  network Q? Impacts correctness for systems with large Q swing on first snapshot.
- [ ] `weight_ptdf`: keep at 0 (backward compat) or remove? → Keep at 0 for now.
- [ ] `B_red_inv` serialization: store in `run_info` dict (as `.tolist()`) alongside saved model.
- [ ] Distributed slack P allocation: use `p_nom` share (same as `pnom_share` feature).
- [ ] **Timing benchmark**: need a benchmark cell to confirm baseline (~0.01–0.1 ms) < DC lpf (~2.6 ms).
