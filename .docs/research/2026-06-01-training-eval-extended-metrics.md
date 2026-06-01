# Research: Extended Metrics in Training Notebook's evaluate_gnn_on_test_set

**Date**: 2026-06-01  
**Context**: Plan 038 adds `p_rmse`, `q_rmse`, `line_flow_rmse`, `line_flow_q0_rmse`, `delta_theta_mae`, `delta_theta_rmse` to the Analysis notebook. This artifact documents where and how the same metrics should be added to the Training notebook so that saved sweep results are self-contained.

## Current State (cell 18, `evaluate_gnn_on_test_set`)

### Metrics currently returned

| Key | Source (line) | Granularity |
|-----|---------------|-------------|
| `vmag_mae` | L238: `np.mean(vmag_errors)` | mean-of-per-snapshot-means |
| `vang_mae` | L239 | same |
| `p_mae` | L240 | same |
| `q_mae` | L241 | same |
| `vmag_rmse` | L242: `sqrt(mean([e**2 for e in vmag_errors]))` | RMSE-of-per-snapshot-means |
| `vang_rmse` | L243 | same |
| `line_flow_mae` | L250: `np.mean(lflow_errors)` | mean-of-per-snapshot-means |
| `line_flow_q0_mae` | L252 | same |
| `line_flow_raw` | L255-262: `{p0_true, p0_pred, q0_true, q0_pred}` | per-element arrays |
| `delta_theta_pred_all` | L296: `np.concatenate(all_delta_theta_pred)` | raw numpy (no metric) |

### Missing keys (needed for plan 038 scatter)

| Key | Data source already in function | Computation |
|-----|--------------------------------|-------------|
| `p_rmse` | `p_errors` list (per-snapshot mean abs error) | `sqrt(mean([e**2 for e in p_errors]))` |
| `q_rmse` | `q_errors` list | same pattern |
| `line_flow_rmse` | `lflow_errors` list | same pattern |
| `line_flow_q0_rmse` | `lflow_q0_errors` list | same pattern |
| `delta_theta_mae` | `all_delta_theta_pred` + `data.y_delta_theta` | per-edge abs error, mean |
| `delta_theta_rmse` | same | per-edge squared error, sqrt mean |

## RMSE Semantic Note

The existing `vmag_rmse`/`vang_rmse` use **RMSE-of-per-snapshot-mean-errors**:
```python
float(np.sqrt(np.mean([e**2 for e in vmag_errors])))
```
where each `e = (vmag_pred - vmag_true).abs().mean().item()` (L153).

This is **not** true element-wise RMSE — it's the RMSE across snapshot-level MAE values. Same semantic as the DC baseline in Analysis. True per-element RMSE would require accumulating per-element squared errors.

**Decision**: Keep the same semantic (RMSE-of-means) for `p_rmse`/`q_rmse`/`line_flow_rmse`/`line_flow_q0_rmse` for consistency with existing `vmag_rmse`/`vang_rmse`. Document this clearly.

For `delta_theta_mae`/`delta_theta_rmse`, raw per-edge arrays are already stored — can compute true element-wise metrics.

## Implementation Touch Points

### 1. `evaluate_gnn_on_test_set` (cell 18)

**Location**: After L248 (closing `}` of metrics dict), before the `if lflow_errors:` block.

**Add p_rmse, q_rmse** — trivial (same pattern as L242-243):
```python
"p_rmse":        float(np.sqrt(np.mean([e**2 for e in p_errors]))),
"q_rmse":        float(np.sqrt(np.mean([e**2 for e in q_errors]))),
```

**Add line_flow_rmse, line_flow_q0_rmse** — after L252 (inside existing `if` guards):
```python
if lflow_errors:
    metrics["line_flow_mae"] = float(np.mean(lflow_errors))
    metrics["line_flow_rmse"] = float(np.sqrt(np.mean([e**2 for e in lflow_errors])))
if lflow_q0_errors:
    metrics["line_flow_q0_mae"] = float(np.mean(lflow_q0_errors))
    metrics["line_flow_q0_rmse"] = float(np.sqrt(np.mean([e**2 for e in lflow_q0_errors])))
```

**Add delta_theta_mae, delta_theta_rmse** — near L296 where `delta_theta_pred_all` is already concatenated. Requires accumulating true values too:

**New accumulator** (near L60):
```python
all_delta_theta_true = []  # [STSI 260601]: Δθ true values for MAE/RMSE
```

**Accumulate** (near L196, inside the same guard):
```python
if angle_mode == "edge_delta" and delta_theta_pred_np is not None:
    all_delta_theta_pred.append(delta_theta_pred_np)
    if hasattr(data, 'y_delta_theta'):
        all_delta_theta_true.append(data.y_delta_theta.cpu().numpy())
```

**Compute** (near L296):
```python
if all_delta_theta_pred and all_delta_theta_true:
    dt_pred = np.concatenate(all_delta_theta_pred)
    dt_true = np.concatenate(all_delta_theta_true)
    dt_err = np.abs(dt_pred - dt_true)
    metrics["delta_theta_mae"] = float(np.mean(dt_err))
    metrics["delta_theta_rmse"] = float(np.sqrt(np.mean(dt_err**2)))
    metrics["delta_theta_pred_all"] = dt_pred  # keep raw for analysis
else:
    if all_delta_theta_pred:
        metrics["delta_theta_pred_all"] = np.concatenate(all_delta_theta_pred)
```

### 2. `save_sweep_results` CSV (cell 19)

**Location**: After L920 (`line_flow_mae`), add new columns:
```python
"line_flow_q0_mae":     tm.get("line_flow_q0_mae"),
"p_rmse":               tm.get("p_rmse"),
"q_rmse":               tm.get("q_rmse"),
"line_flow_rmse":       tm.get("line_flow_rmse"),
"line_flow_q0_rmse":    tm.get("line_flow_q0_rmse"),
"delta_theta_mae":      tm.get("delta_theta_mae"),
"delta_theta_rmse":     tm.get("delta_theta_rmse"),
```

### 3. `plot_all_runs_accuracy_metrics` (cell 19) — optional

Currently plots bars for: `vmag_mae`, `vang_rmse`, `p_mae`, `q_mae`, `line_flow_mae`, `line_flow_q0_mae`.

Could optionally add `delta_theta_mae` as a 7th subplot. Low priority — the main scatter lives in Analysis.

## Analysis Notebook Compatibility

When the Training notebook saves these keys in `test_metrics`, the Analysis notebook's `check_hparam_results` will find them via `run["test_metrics"].get("p_rmse")` without needing to re-evaluate. This means:

- **Fresh sweep runs** (post-implementation): scatter works immediately.
- **Old saved runs** (pre-implementation): will have `None` for new keys; `.get()` returns `None` → NaN → filtered from scatter subplots. Acceptable — user can re-evaluate with updated code if needed.

## Dependencies

- None external. All raw data is already collected in the function.
- `data.y_delta_theta` attribute must exist on the dataset (it does when `angle_mode="edge_delta"` and `PowerFlowDataset` includes it).

## Scope

| Change | Effort | Priority |
|--------|--------|----------|
| `p_rmse`, `q_rmse` in metrics dict | 2 lines | High — trivial |
| `line_flow_rmse`, `line_flow_q0_rmse` | 2 lines | High — trivial |
| `delta_theta_mae`, `delta_theta_rmse` | ~10 lines (accumulator + compute) | High — key metric |
| CSV columns in `save_sweep_results` | 7 lines | Medium — only matters for new sweeps |
| `plot_all_runs_accuracy_metrics` bar | Optional | Low |

## Recommendation

Implement touch points 1 and 2 (cell 18 metrics + cell 19 CSV). Skip touch point 3 (bar chart) — the timing-vs-accuracy scatter in Analysis is the primary visualization.

Total addition: ~20 lines of code across 2 cells.
