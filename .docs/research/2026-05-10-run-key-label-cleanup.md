# Research: Run Key / Legend Label / Style Cleanup

## Goal
Identify dead code, missing new-mode support, and inconsistencies in `make_run_key`, `build_legend_label`, `_get_run_styles`, `PhysicsConfig.label()`, and all their consumers across both V2.6 Training and Analysis notebooks.

## Key Findings

### 1. `PhysicsConfig.label()` includes dead/always-on flags
- **Location**: Training cell 12 (lines ~1431–1443)
- **What**: `label()` appends `"PB"` when `use_power_balance=True` and `"AR"` when `use_angle_ref_penalty=True`. Every config in every sweep always sets `use_power_balance=True`. Only ~half set `use_angle_ref_penalty=True`, but the flag has been a no-op in `with_encoder` mode (angle ref is structurally enforced). `"PB"` is always present → noise in keys; `"AR"` is sometimes present but rarely the distinguishing factor.
- **Why it matters**: The label string flows into `make_run_key` (via `pcfg.label()`), `run_info["physics_label"]`, and `build_legend_label`. "PB" and "AR" clutter labels without adding information.

### 2. `make_run_key` — Training only, no Analysis equivalent
- **Location**: Training cell 19, `def make_run_key(...)` (line ~3571)
- **What**: Produces stable dedup key for checkpoint resume. Includes `pcfg.label()`, all hparams, `head_mode`, `use_pnom_share`, `conv_mode`, `drop_rate`.
- **Why it matters**: The key format is *correct and up-to-date* (already has all new modes). However, the **legacy backfill** inside `run_hparam_sweep`'s checkpoint resume (line ~3252) builds a *different* format — it omits `conv_mode`, `drop_rate`, and `use_pnom_share`, and uses `f"_head{head_modes}"` instead of `f"hm{head_mode}"`. Old runs could collide with new ones.

### 3. `build_legend_label` — Training version is MISSING new modes
- **Location**: Training cell 19, `def build_legend_label(...)` (line ~3659)
- **What**: The `varying` list checks these keys:
  ```
  physics_mode, physics_label, weight_ptdf, ptdf_loss_mode,
  ptdf_branch_mode, ptdf_alpha, batch_size, lr, conv_type,
  y_matrix_source, warmup_epochs, hidden_dim, num_layers
  ```
  **Missing from Training**: `head_mode`, `conv_mode`, `drop_rate`, `use_pnom_share`/`n_node_features`.
  If two runs differ only in `head_mode` or `conv_mode`, they get *identical* legend labels → indistinguishable plots.
- **Why it matters**: This is a live bug when sweeping `head_modes`, `conv_modes`, or `drop_rates`.

### 4. `build_legend_label` — Analysis version is PARTIALLY updated
- **Location**: Analysis cell 18, `def build_legend_label(...)` (line ~2576)
- **What**: The Analysis `varying` list includes `head_mode` and `n_node_features`, **but NOT** `conv_mode` or `drop_rate`. And it uses `n_node_features` (a key that only exists if backfilled/added manually) instead of `use_pnom_share` or `node_feature_mode`.
  Also adds formatting:
  ```python
  if "conv_mode" in varying:
      parts.append(f"cm={r.get('conv_mode', 'old')}")
  if "head_mode" in varying:
      parts.append(f"hm={r.get('head_mode', 'standard')}")
  if "n_node_features" in varying:
      parts.append(f"nf={r.get('n_node_features', 8)}")
  ```
  But `conv_mode` is NOT in the `varying` candidate list above — it's in the label part but will never trigger because it's not scanned.
- **Why it matters**: The Analysis label code *looks* updated but `conv_mode` can never appear as varying, so it's unreachable.

### 5. `_get_run_styles` — Training version MISSING new modes
- **Location**: Training cell 19, `def _get_run_styles(...)` (line ~3629)
- **What**: Scans only: `physics_mode`, `physics_label`, `weight_ptdf`, `batch_size`, `lr`, `conv_type`, `y_matrix_source`. Missing: `head_mode`, `conv_mode`, `drop_rate`.
- **Analysis version**: Scans `physics_mode`, `physics_label`, `weight_ptdf`, `batch_size`, `lr`, `conv_type`, `head_mode`, `n_node_features`, `y_matrix_source`. Has `head_mode` and `n_node_features` but missing `conv_mode`, `drop_rate`.
- **Why it matters**: Visual style assignment doesn't distinguish new modes → runs get same color/linestyle.

### 6. `run_info` dict — contains dead fields
- **Location**: Training cell 19, `run_info = {...}` (line ~3468)
- **What**: Always stores `use_power_balance`, `use_angle_ref_penalty`, `use_q_partial_mode` even though:
  - `use_power_balance` is always `True`
  - `use_angle_ref_penalty` is always `False` or `True` but the actual loss function already encodes this via `PhysicsConfig`
  - These are already encoded in `physics_label` (e.g. "PB", "AR" suffixes)
- **Why it matters**: Redundant fields inflate run dicts and saved JSON/PKL without adding queryable information. They're never used as discriminators in sweeps.

### 7. `save_sweep_results` CSV — missing new mode columns
- **Location**: Training cell 19, inside `save_sweep_results()`
- **What**: CSV includes `use_power_balance`, `use_angle_ref`, `use_q_partial`, but does NOT include `head_mode`, `conv_mode`, `drop_rate`, `use_pnom_share`, `node_feature_mode`.
- **Why it matters**: Analysis can't distinguish new runs from CSV alone.

### 8. `save_sweep_results` manifest — missing new mode summaries
- **Location**: Training cell 19, manifest JSON
- **What**: `hyperparams` dict includes `physics_labels`, `w_phys`, `w_ptdf`, `batch_sizes`, `learning_rates`, `conv_types`, `ptdf_modes`, `warmup_epochs`, `seeds` — but NOT `head_modes`, `conv_modes`, `drop_rates`.
- **Why it matters**: Manifest summary is incomplete for new sweeps.

### 9. `print_sweep_summary` — hardcoded columns, no new modes
- **Location**: Both notebooks, `def print_sweep_summary(...)`
- **What**: Header is `Rank, w_phys, w_ptdf, bs, lr, Train, Val, Test, Time(s)`. Doesn't show `head_mode`, `conv_mode`, `physics_mode`, etc.
- **Why it matters**: Summary table can't distinguish runs that differ on new modes.

### 10. `create_comparison_dataframe` — missing new mode columns
- **Location**: Training cell 19, `def create_comparison_dataframe(...)`
- **What**: Only includes `tag`, `w_phys`, `w_ptdf`, `bs`, `lr`, losses, training_time. No head_mode, conv_mode, drop_rate, physics_mode.
- **Why it matters**: Same as CSV — comparison doesn't capture new dimensions.

### 11. Legacy backfill key format mismatch
- **Location**: Training cell 19, inside `run_hparam_sweep()` resume block
- **What**: For old runs without `run_key`, builds:
  ```python
  f"{physics_label}_bs{bs}_lr{lr:.2e}_h{hd}_L{nl}_{ct}_{ys}_wu{we}_s{seed}_head{head_modes}"
  ```
  While `make_run_key` builds:
  ```python
  f"{pcfg.label()}_bs{bs}_lr{lr:.2e}_h{hd}_L{nl}_{ct}_{ys}_wu{we}_s{seed}_hm{head_mode}_pn{int(pnom)}_cm{conv_mode}_dr{drop_rate}"
  ```
  Format mismatch: `_head{x}` vs `_hm{x}_pn{y}_cm{z}_dr{w}`. This causes legacy runs to never match new run keys, which is actually *correct* for dedup — but it means old runs also can't be "resumed" into a new sweep with changed key format.
- **Why it matters**: Minor — works by accident but is fragile. Should use `make_run_key` directly.

### 12. `physics_mode` in `_get_run_styles` / `build_legend_label` — dead discriminator
- **What**: `physics_mode` is either `"rich"` or `"simple"`. In practice, ALL current configs use `"rich"` — `"simple"` mode was an early experiment. It's never varied in sweeps.
- **Why it matters**: `physics_mode` takes up a slot in the varying-key list but never produces variance. Could be demoted.

## Patterns to Follow
| Pattern | Example Location | Notes |
|---------|------------------|-------|
| `varying` auto-detection | `build_legend_label`, `_get_run_styles` | Same list of keys must be checked in both |
| Run key via `make_run_key` | Training cell 19 | All varying params must be in key |
| `run_info` metadata | Training cell 19 | Fields stored here flow to CSV/JSON/manifest |

## Key Files
| File | Purpose | Relevance |
|------|---------|-----------|
| Training cell 19 | `make_run_key`, `build_legend_label`, `_get_run_styles`, `create_comparison_dataframe`, `save_sweep_results`, `print_sweep_summary`, `run_hparam_sweep` (resume backfill) | All Training-side key/label/style + save logic |
| Training cell 12 | `PhysicsConfig.label()`, `physics_informed_loss_batch` | Label generation, dead flags |
| Analysis cell 18 | `build_legend_label`, `_get_run_styles`, `print_sweep_summary` | Analysis-side label/style logic |

## Summary of Changes Needed

### A. Dead code to remove or simplify
1. **`PhysicsConfig.label()`**: Remove `"PB"` suffix (always on). Consider removing `"AR"` suffix (rarely meaningful, structural no-op in encoder mode).
2. **`run_info` dict**: Remove `use_power_balance` and `use_angle_ref_penalty` keys (always True/always encoded elsewhere). Keep `use_q_partial_mode` (it's off by default and occasionally toggled).
3. **`save_sweep_results` CSV**: Remove `use_power_balance` and `use_angle_ref` columns.
4. **`physics_mode`** in varying lists: Consider demoting priority (always "rich" in practice).

### B. New modes to add
1. **`build_legend_label` (Training)**: Add `head_mode`, `conv_mode`, `drop_rate` to both the `varying` candidate list and the label-building `if` blocks.
2. **`build_legend_label` (Analysis)**: Add `conv_mode` and `drop_rate` to the `varying` candidate list (they're in the label part but unreachable). Replace `n_node_features` with `node_feature_mode` or `use_pnom_share` for consistency.
3. **`_get_run_styles` (Training)**: Add `head_mode`, `conv_mode`, `drop_rate` to scanned keys.
4. **`_get_run_styles` (Analysis)**: Add `conv_mode`, `drop_rate` to scanned keys.
5. **`save_sweep_results` CSV**: Add `head_mode`, `conv_mode`, `drop_rate`, `use_pnom_share`, `node_feature_mode`.
6. **`save_sweep_results` manifest**: Add `head_modes`, `conv_modes`, `drop_rates` to hyperparams summary.
7. **`print_sweep_summary` (both)**: Add `head_mode`, `conv_mode` columns.
8. **`create_comparison_dataframe`**: Add `head_mode`, `conv_mode`, `drop_rate`, `physics_mode`.
9. **Legacy backfill key**: Use `make_run_key` directly (construct a dummy PhysicsConfig from stored fields).

## Constraints & Considerations
- **Backward compat**: Old saved run PKL/JSON files won't have `head_mode`, `conv_mode`, `drop_rate` keys. All `.get()` calls need defaults: `head_mode="standard"`, `conv_mode="old"`, `drop_rate=0.1`, `use_pnom_share=False`.
- **`n_node_features`** in Analysis is a quirk — no run_info naturally has this key. It was likely added via manual backfill. Should be replaced with `node_feature_mode` (which IS in run_info).
- **Removing `PB`/`AR` from `PhysicsConfig.label()`** changes run keys for all past and future runs. This is a KEY FORMAT CHANGE — existing checkpoint files will fail to resume. Options: (a) accept it (re-run from scratch), or (b) add a format version to keys.

## Resolved Questions (2026-05-10)
- [x] Remove `use_q_partial_mode` from `PhysicsConfig.label()` — YES, always False.
- [x] Remove `physics_mode` from label — YES, always "rich".
- [x] Accept that changing `PhysicsConfig.label()` breaks checkpoint resume — YES, OK.
