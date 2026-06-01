# Research: delta_theta Scatter/Histogram Range and Plot Structure

## Goal
Assess why the delta_theta scatter/histogram/error-by-range plots in `check_hparam_results` show values spanning the same range as absolute bus angles (up to ~-50°), and why these plots appear as separate figures instead of integrated into the existing multi-panel layout.

---

## Key Findings

### 1. Data Extraction Path Is Correct (No Computation Bug)

**Ground truth** (`dt_true`):
- **Location**: `GNN_Powerflow_V2.6_Analysis.ipynb`, cell 10, L288–293
- Computed as: `theta_true[edge_index_fwd[0]] - theta_true[edge_index_fwd[1]]` where `theta_true = y[:, 1]` (bus angle in **radians** from `network.buses_t.v_ang`)
- `edge_index_fwd` = `edge_index[:, fwd_mask]` where `fwd_mask` is `[True, False, True, False, ...]` (every even-indexed edge) when `ptdf_branch_mode="all"`
- This IS the correct per-edge angle difference θ_from − θ_to in radians

**Predictions** (`dt_pred`):
- **Location**: cell 9 (PowerFlowGNN), L367–374
- Model selects forward edges via `all_fwd[::2] = True` → `h_edges_fwd = h_edges[all_fwd]`
- Passes through linear head: `delta_theta_pred = self.edge_angle_pred(h_edges_fwd).squeeze(-1)` → shape `[E_fwd]`
- Returned as `out[2]` from `model(data)` (3-tuple: `node_pred, ptdf_pred, delta_theta_pred`)

**Accumulation** in `predict_network_results_with_masks` (cell 12, L170–174):
```python
delta_theta_pred = out[2] if len(out) > 2 else None
if collect_delta_theta and delta_theta_pred is not None:
    _dt_pred_rows.append(delta_theta_pred.cpu().numpy())
    if hasattr(data, "y_delta_theta") and data.y_delta_theta is not None:
        _dt_true_rows.append(data.y_delta_theta.cpu().numpy())
```

**Collection** in `_collect_test_set_predictions` (cell 23, L53–69):
- Per-network `dt_data["dt_pred"]` and `dt_data["dt_true"]` appended, then `np.concatenate`ed
- Final shapes: `[total_fwd_edges × n_snapshots × n_networks]` (1D)

**Edge ordering consistency verified**:
- Dataset: edges stored as `[fwd_0, rev_0, fwd_1, rev_1, ...]`, `fwd_mask` selects even indices
- Model: `h_edges[::2]` selects the same even-indexed positions
- Both produce `E_fwd = n_lines + n_trafos` values per snapshot

### 2. Root Cause CONFIRMED: Bad PF Solutions with Outlier Bus Angles

**Diagnostic results** (2026-06-01):

| Network | Buses | Bus v_ang range | Δθ range | |Δθ|>20° |
|---------|-------|-----------------|----------|---------|
| 0 | 58 | [-23.8°, 0.0°] | [-7.6°, 12.8°] | 0% |
| 1 | 57 | [-22.2°, 0.7°] | [-5.9°, 9.4°] | 0% |
| 2 | 58 | [-13.8°, **+65.6°**] | [**-54.3°**, 9.8°] | 1.2% |

**Network 2 has a single bus at +62.3°** while all other buses are in [-14°, 0°]. This is a non-physical PF solution that passed the existing sanity checks. The edge connecting this outlier bus to its neighbour produces the -54° delta.

Manual vs stored check: `match: True` for all networks — the computation is correct, the data is bad.

**Conclusion**: The y_delta_theta computation and plot extraction are correct. The problem is upstream: `sanity_check_power_flow` in DataGen doesn't filter networks with non-physical bus angles / edge deltas.

### 3. Possible Diagnostic to Confirm (vs an Actual Bug)

If the scatter plot shows the **bulk** of delta_theta values at ±40–50° (not just outliers), there may be a latent bug. Recommend running this diagnostic:

```python
# In a cell after data collection:
for lbl, data in all_preds.items():
    if "dt_true" in data:
        dt_deg = data["dt_true"] * (180/np.pi)
        print(f"{lbl}: dt_true range [{dt_deg.min():.1f}°, {dt_deg.max():.1f}°], "
              f"mean={dt_deg.mean():.2f}°, std={dt_deg.std():.2f}°, "
              f"|Δθ|>20°: {(np.abs(dt_deg)>20).sum()}/{len(dt_deg)} "
              f"({100*(np.abs(dt_deg)>20).mean():.1f}%)")
```

If >10% of edges have |Δθ| > 20°, the training data is genuinely stressed. If the bulk is at large values, investigate whether `edge_index_fwd` is accidentally using non-adjacent bus pairs.

### 4. Plot Structure: Delta Theta Exists as Separate Standalone Functions

**Current structure** (cell 17, L920–928 in `check_hparam_results`):
```python
# Main plots (integrated multi-panel figures):
plot_full_test_scatter(all_preds, ...)       # 4 bus + 2 line panels
plot_error_histograms(all_preds, ...)        # 4 bus + 2 line panels  
plot_errors_vs_operating_range(all_preds, ...)  # 4 separate figures (1 per bus qty)

# Delta theta plots (SEPARATE standalone figures, called after main plots):
plot_delta_theta_scatter(all_preds, ...)         # 1 panel per model
plot_delta_theta_histogram(all_preds, ...)       # 1 panel per model
plot_delta_theta_vs_operating_range(all_preds, ...)  # 1 panel per model
```

**Problem**: Delta theta plots appear as separate figures AFTER the main scatter/histogram/error plots finish. They're not integrated into the same multi-panel layout.

**Function locations**:
| Function | Cell | Lines | Panels |
|----------|------|-------|--------|
| `plot_full_test_scatter` | 23 | L74–98 | 4 bus + 2 line in one row |
| `plot_error_histograms` | 23 | L101–135 | 4 bus + 2 line in one row |
| `plot_errors_vs_operating_range` | 24 | L0–196 | 1 figure per bus qty, stacked subplots per model |
| `plot_delta_theta_scatter` | 23 | L136–155 | 1 standalone panel per model |
| `plot_delta_theta_histogram` | 23 | L158–178 | 1 standalone panel per model |
| `plot_delta_theta_vs_operating_range` | 24 | L197–320 | 1 standalone figure per model |

---

## Patterns to Follow

| Pattern | Example | Notes |
|---------|---------|-------|
| Multi-panel scatter | `plot_full_test_scatter` (cell 23, L74–98) | 4+2 panels in single figure; add Δθ as 7th panel |
| Multi-panel histogram | `plot_error_histograms` (cell 23, L101–135) | Same structure; add Δθ as 7th panel |
| Stacked error-vs-range | `plot_errors_vs_operating_range` (cell 24, L0–196) | 1 figure per quantity, N models stacked; add 5th figure for Δθ |

---

## Key Files

| File | Cell | Purpose | Relevance |
|------|------|---------|-----------|
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 10 | `PowerFlowDataset._create_graph_data` | Computes `y_delta_theta` from `theta_true[from] - theta_true[to]` |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 9 | `PowerFlowGNN.forward` | Produces `delta_theta_pred` via edge angle head |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 12 | `predict_network_results_with_masks` | Collects dt_pred/dt_true per snapshot |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 23 | `_collect_test_set_predictions` | Aggregates across networks; stores in `all_preds` |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 23 | `plot_delta_theta_scatter/histogram` | Standalone Δθ plot functions |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 24 | `plot_delta_theta_vs_operating_range` | Standalone Δθ error-vs-range |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | 17 | `check_hparam_results` L920–928 | Calls all plot functions |

---

## Constraints & Considerations

- **Units**: `dt_pred`/`dt_true` stored in RADIANS; plot functions multiply by `RAD2DEG = 180/π`
- **Alignment**: Model's `h_edges[::2]` and dataset's `edge_index[:, fwd_mask]` are consistent (both select even-indexed/forward edges)
- **Shape**: Both have length `E_fwd` per snapshot; concatenated across snapshots/networks
- **ptdf_branch_mode**: Must be `"all"` for edge_delta models (includes transformers for BFS connectivity); enforced in `predict_network_results_with_masks` L125

---

## Recommendations

### R1: Run diagnostic to distinguish "stressed data" from "bug"
Add a diagnostic print in `check_hparam_results` after `all_preds` is collected:
- Print min/max/mean/std of `dt_true` (in degrees)
- Print fraction of edges with |Δθ| > 20°
- If the distribution is heavily tailed (most values <15°, few outliers at 40–50°), it's genuine data variation
- If the BULK is at 40–50°, investigate `edge_index_fwd` construction

### R2: Integrate delta_theta into existing multi-panel plots
**Scatter**: Extend `plot_full_test_scatter` to add a 7th panel for Δθ (degrees) when `"dt_pred"` is available in `all_preds[label]`. Signature: `n_panels = 7 if has_dt else (6 if has_lf else 4)`.

**Histogram**: Extend `plot_error_histograms` with same approach — 7th panel for Δθ error.

**Error-vs-range**: Add Δθ as a 5th quantity in `col_specs` (or as a separate figure, same layout). Since Δθ is edge-level (different array shape from bus-level), it needs its own data source: `data["dt_pred"] - data["dt_true"]` rather than `data["pred"][:,col] - data["true"][:,col]`.

### R3: Remove standalone delta_theta plot functions after integration
Once integrated, remove `plot_delta_theta_scatter`, `plot_delta_theta_histogram`, `plot_delta_theta_vs_operating_range` and their call sites (L926–928).

### R4: Consider clipping outliers in scatter for readability
If the diagnosis confirms outliers are genuine but rare, add optional `clip_percentile` to limit axis range (e.g., 1st–99th percentile) so the bulk of data is visible.

---

## Open Questions

- [ ] What fraction of `dt_true` values exceed ±20°? (diagnostic needed)
- [ ] Is the sign distribution of `dt_true` roughly symmetric, or biased negative? (would indicate CSV bus0/bus1 ordering bias)
- [ ] Should the integrated Δθ panel use the same scale as V_ang, or auto-scale independently?
