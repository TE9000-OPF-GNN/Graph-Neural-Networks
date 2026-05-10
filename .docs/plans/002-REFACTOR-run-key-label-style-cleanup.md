---
type: REFACTOR
status: todo
priority: Medium
effort: 2h
labels: [training, analysis, sweep-infrastructure]
depends-on: []
created: 2026-05-10
completed:
summary: ""
---

# Clean Up Run Key, Legend Label, and Style Infrastructure

## Problem
After adding `head_mode`, `conv_mode`, `drop_rate`, and `use_pnom_share` to the sweep pipeline, the supporting functions (`make_run_key`, `build_legend_label`, `_get_run_styles`, `PhysicsConfig.label()`, `save_sweep_results`, `print_sweep_summary`, `create_comparison_dataframe`) were not fully updated. This causes:
1. **Indistinguishable runs** in plots when only new modes vary (live bug)
2. **Dead code** for always-on flags (`use_power_balance`, `use_angle_ref_penalty`, `physics_mode`, `use_q_partial_mode`)
3. **Missing columns** in CSV export and summary tables for new modes
4. **Inconsistencies** between Training and Analysis notebooks

## Solution
Single pass through both notebooks to: remove dead discriminators, add new mode support to all key/label/style/save functions, and sync Training ↔ Analysis.

**Design decisions:**
- Remove `PB`, `AR`, `Qp`, `mode-{x}` from `PhysicsConfig.label()` — these are always-on or always-off, adding noise. Label becomes just `f"w-{w_phys}"` plus PTDF parts. **This breaks checkpoint resume for old saved runs — accepted.**
- Remove `use_power_balance`, `use_angle_ref_penalty`, `use_q_partial_mode` from `run_info` dict — redundant with `physics_label`.
- Replace `n_node_features` in Analysis with `node_feature_mode` (which exists in `run_info`).
- Use `make_run_key` directly in legacy backfill instead of a hand-rolled format.

## Scope
**Included:**
- `PhysicsConfig.label()` simplification
- `make_run_key` — no changes needed (already up-to-date), but fix legacy backfill
- `build_legend_label` — add new modes (both notebooks)
- `_get_run_styles` — add new modes (both notebooks)
- `save_sweep_results` CSV — add new columns, remove dead columns
- `save_sweep_results` manifest — add new hparam summaries
- `print_sweep_summary` — add new mode columns (both notebooks)
- `create_comparison_dataframe` — add new mode columns
- `run_info` dict — remove dead fields, keep new ones

**Not Included:**
- Changes to `PhysicsConfig` fields themselves (keep the dataclass fields, just don't surface them in `label()`)
- Changes to `physics_informed_loss_batch` logic
- Changes to the Analysis `backfill_*` functions
- DataGen notebook changes

## Affected Files
| File | Change |
|------|--------|
| `GNN_Powerflow_V2.6_Training.ipynb` cell 12 | Simplify `PhysicsConfig.label()` |
| `GNN_Powerflow_V2.6_Training.ipynb` cell 19 | Update `build_legend_label`, `_get_run_styles`, `create_comparison_dataframe`, `save_sweep_results`, `print_sweep_summary`, `run_info` dict, legacy backfill |
| `GNN_Powerflow_V2.6_Analysis.ipynb` cell 18 | Update `build_legend_label`, `_get_run_styles`, `print_sweep_summary` |

## Implementation Steps

### 1. Simplify `PhysicsConfig.label()` (Training cell 12)

**Before:**
```python
    def label(self) -> str:
        parts = [f"mode-{self.physics_mode}", f"w-{self.w_phys}"]
        if self.use_power_balance:
            parts.append("PB")
        if self.use_angle_ref_penalty:
            parts.append("AR")
        if self.use_q_partial_mode:
            parts.append("Qp")
        if self.use_ptdf_loss:
            ptdf_part = f"PTDF-{self.ptdf_loss_mode}-pt{self.weight_ptdf}-{self.ptdf_branch_mode}"
            if self.ptdf_loss_mode == "mixed":
                ptdf_part += f"-a{self.ptdf_alpha}"
            parts.append(ptdf_part)
        return "_".join(parts)
```

**After:**
```python
    def label(self) -> str:
        parts = [f"w-{self.w_phys}"]
        if self.use_ptdf_loss:
            ptdf_part = f"PTDF-{self.ptdf_loss_mode}-pt{self.weight_ptdf}-{self.ptdf_branch_mode}"
            if self.ptdf_loss_mode == "mixed":
                ptdf_part += f"-a{self.ptdf_alpha}"
            parts.append(ptdf_part)
        return "_".join(parts)
```

### 2. Remove dead fields from `run_info` dict (Training cell 19)

Remove these three lines from the `run_info` dict inside `run_hparam_sweep`:
```python
            "use_power_balance": pcfg.use_power_balance,
            "use_angle_ref_penalty": pcfg.use_angle_ref_penalty,
            "use_q_partial_mode": pcfg.use_q_partial_mode,
```

### 3. Fix legacy backfill in `run_hparam_sweep` checkpoint resume (Training cell 19)

**Before:**
```python
                r["run_key"] = (
                    f"{physics_label}_"
                    f"bs{bs}_"
                    f"lr{lr:.2e}_"
                    f"h{hidden_dim}_"
                    f"L{num_layers}_"
                    f"{conv_type}_"
                    f"{ysrc}_"
                    f"wu{warmup_epochs}_"
                    f"s{seed}"
                    f"_head{head_modes}"
                )
```

**After:**
```python
                r["run_key"] = (
                    f"{physics_label}_"
                    f"bs{bs}_"
                    f"lr{lr:.2e}_"
                    f"h{hidden_dim}_"
                    f"L{num_layers}_"
                    f"{conv_type}_"
                    f"{ysrc}_"
                    f"wu{warmup_epochs}_"
                    f"s{seed}_"
                    f"hm{r.get('head_mode', 'standard')}_"
                    f"pn{int(r.get('use_pnom_share', False))}_"
                    f"cm{r.get('conv_mode', 'old')}_"
                    f"dr{r.get('drop_rate', 0.1)}"
                )
```

### 4. Update `_get_run_styles` — Training (Training cell 19)

**Before:**
```python
    varying = [
        k for k in ("physics_mode", "physics_label", "weight_ptdf",
                    "batch_size", "lr", "conv_type", "y_matrix_source")
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

**After:**
```python
    varying = [
        k for k in ("physics_label", "weight_ptdf",
                    "head_mode", "conv_mode", "drop_rate",
                    "batch_size", "lr", "conv_type", "y_matrix_source")
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

### 5. Update `_get_run_styles` — Analysis (Analysis cell 18)

**Before:**
```python
    varying = [
        k for k in ("physics_mode", "physics_label", "weight_ptdf",
                    "batch_size", "lr", "conv_type", "head_mode", "n_node_features", "y_matrix_source")
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

**After:**
```python
    varying = [
        k for k in ("physics_label", "weight_ptdf",
                    "head_mode", "conv_mode", "drop_rate",
                    "batch_size", "lr", "conv_type", "y_matrix_source")
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

### 6. Update `build_legend_label` — Training (Training cell 19)

Add new keys to both the `varying` candidate list and the label-building blocks.

**Before (varying list):**
```python
    varying = [
        k for k in [
            "physics_mode",
            "physics_label",
            "weight_ptdf",
            "ptdf_loss_mode",
            "ptdf_branch_mode",
            "ptdf_alpha",
            "batch_size",
            "lr",
            "conv_type",
            "y_matrix_source",
            "warmup_epochs",
            "hidden_dim",
            "num_layers",
        ]
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

**After (varying list):**
```python
    varying = [
        k for k in [
            "physics_label",
            "weight_ptdf",
            "ptdf_loss_mode",
            "ptdf_branch_mode",
            "ptdf_alpha",
            "head_mode",
            "conv_mode",
            "drop_rate",
            "batch_size",
            "lr",
            "conv_type",
            "y_matrix_source",
            "warmup_epochs",
            "hidden_dim",
            "num_layers",
        ]
        if len({str(x.get(k)) for x in runs}) > 1
    ]
```

**Before (label building — after the `conv_type` block, before `return`):**
```python
    if "conv_type" in varying:
        parts.append(r.get("conv_type", ""))

    return " | ".join(parts)
```

**After:**
```python
    if "conv_type" in varying:
        parts.append(r.get("conv_type", ""))
    if "head_mode" in varying:
        parts.append(f"hm={r.get('head_mode', 'standard')}")
    if "conv_mode" in varying:
        parts.append(f"cm={r.get('conv_mode', 'old')}")
    if "drop_rate" in varying:
        parts.append(f"dr={r.get('drop_rate', 0.1)}")

    return " | ".join(parts)
```

Also remove the `physics_mode` block:
```python
    # REMOVE this block:
    if "physics_mode" in varying:
        parts.append(r.get("physics_mode", "rich"))
```

### 7. Update `build_legend_label` — Analysis (Analysis cell 18)

Same changes as Training step 6. Additionally replace `n_node_features` with `node_feature_mode`:

**Before:**
```python
            "head_mode",
            "n_node_features",
            "y_matrix_source",
```

**After:**
```python
            "head_mode",
            "conv_mode",
            "drop_rate",
            "y_matrix_source",
```

And in the label-building section, remove the `n_node_features` block and `physics_mode` block, add `conv_mode` and `drop_rate` blocks (same as Training).

### 8. Update `save_sweep_results` CSV columns (Training cell 19)

**Add** after the existing `y_matrix_source` row:
```python
                "head_mode":            r.get("head_mode", "standard"),
                "conv_mode":            r.get("conv_mode", "old"),
                "drop_rate":            r.get("drop_rate", 0.1),
                "use_pnom_share":       r.get("use_pnom_share", False),
                "node_feature_mode":    r.get("node_feature_mode", "base"),
```

**Remove** these three rows:
```python
                "use_power_balance":    r.get("use_power_balance"),
                "use_angle_ref":        r.get("use_angle_ref_penalty"),
                "use_q_partial":        r.get("use_q_partial_mode"),
```

### 9. Update `save_sweep_results` manifest (Training cell 19)

**Add** to the `hyperparams` dict:
```python
            "head_modes":     sorted(set(r.get("head_mode", "standard") for r in runs)),
            "conv_modes":     sorted(set(r.get("conv_mode", "old") for r in runs)),
            "drop_rates":     sorted(set(r.get("drop_rate", 0.1) for r in runs)),
```

### 10. Update `print_sweep_summary` (both notebooks)

**Before:**
```python
    print(f"{'Rank':<5}{'w_phys':<9}{'w_ptdf':<9}{'bs':<6}{'lr':<11}"
          f"{'Train':<11}{'Val':<11}{'Test':<11}{'Time(s)':<9}")
    print("-" * 92)
    for i, run in enumerate(sorted_runs[:top_n], 1):
        tm = run["test_metrics"]
        print(
            f"{i:<5}{run['weight_physics']:<9}{run['weight_ptdf']:<9}"
            f"{run['batch_size']:<6}{run['lr']:<11.2e}"
            f"{run['final_train_loss']:<11.6f}{run['final_val_loss']:<11.6f}"
            f"{tm['test_total']:<11.6f}{run['training_time']:<9.1f}"
        )
```

**After:**
```python
    print(f"{'Rank':<5}{'w_phys':<9}{'w_ptdf':<9}{'head':<14}{'conv':<14}"
          f"{'bs':<6}{'lr':<11}"
          f"{'Train':<11}{'Val':<11}{'Test':<11}{'Time(s)':<9}")
    print("-" * 120)
    for i, run in enumerate(sorted_runs[:top_n], 1):
        tm = run["test_metrics"]
        print(
            f"{i:<5}{run['weight_physics']:<9}{run['weight_ptdf']:<9}"
            f"{run.get('head_mode','standard'):<14}{run.get('conv_mode','old'):<14}"
            f"{run['batch_size']:<6}{run['lr']:<11.2e}"
            f"{run['final_train_loss']:<11.6f}{run['final_val_loss']:<11.6f}"
            f"{tm['test_total']:<11.6f}{run['training_time']:<9.1f}"
        )
```

### 11. Update `create_comparison_dataframe` (Training cell 19)

**Add** to the `rows.append({...})` dict:
```python
            "head_mode":     r.get("head_mode", "standard"),
            "conv_mode":     r.get("conv_mode", "old"),
            "drop_rate":     r.get("drop_rate", 0.1),
            "physics_label": r.get("physics_label", ""),
```

## Acceptance Criteria
- [ ] `PhysicsConfig.label()` output no longer contains `PB`, `AR`, `Qp`, or `mode-` prefixes
- [ ] `build_legend_label` distinguishes runs that differ only on `head_mode`, `conv_mode`, or `drop_rate` (in both notebooks)
- [ ] `_get_run_styles` assigns different colors/styles when `head_mode`, `conv_mode`, or `drop_rate` vary (in both notebooks)
- [ ] `save_sweep_results` CSV includes `head_mode`, `conv_mode`, `drop_rate`, `use_pnom_share`, `node_feature_mode` columns and does NOT include `use_power_balance`, `use_angle_ref`, `use_q_partial`
- [ ] `print_sweep_summary` shows `head_mode` and `conv_mode` columns (both notebooks)
- [ ] `run_info` dict no longer contains `use_power_balance`, `use_angle_ref_penalty`, `use_q_partial_mode`
- [ ] Training and Analysis notebook `build_legend_label` and `_get_run_styles` have the same candidate key lists
- [ ] Old runs without new keys load gracefully (all `.get()` calls have defaults)
