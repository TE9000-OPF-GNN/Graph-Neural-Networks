# Research: Extra Bus Generation Not Working in V2.5.2

## Goal
Understand why `generate_training_data_with_topology` with `extra_bus_prob > 0` produces networks with a fixed bus count (always 30 for IEEE30) instead of varying bus counts (30–33) as the V2.4-era dataset demonstrates.

## Key Findings

### 1. Indentation Bug in `create_csv_based_topology_variant`
- **Location**: `GNN_Powerflow_V2.5.2.ipynb`, Cell 22 (`create_csv_based_topology_variant`), source lines 298–373
- **What it does**: Section 3 ("Optionally add extra buses + lines") is indented 8 spaces, nesting it inside the Section 2c `if trafo_modification_prob > 0.0` block (line 281, 4 spaces). When `trafo_modification_prob` is 0.0 (its default), the entire extra-bus block is **silently skipped**.
- **Why it matters**: This is the root cause. The V2.5.2 call at Cell 85 passes `extra_bus_prob=0.5` but does NOT pass `trafo_modification_prob`, so it defaults to 0.0, and the extra bus code never executes.

### 2. The Bug Is Present in ALL Notebook Versions
- **Verified in**: V2.4, V2.4.1, V2.4_tempfile1, V2.4_tempfile2, V2.4.backup 1, V2.5, V2.5.1, V2.5.2
- **All show identical indentation**: comment at 4 spaces, `if extra_bus_prob` at 8 spaces
- **Implication**: The V2.4 dataset (`ieee30_large`, timestamp `20260328_014317`) with bus counts `[30, 31, 32, 33]` was likely generated when the code was in a transient state with correct indentation, or generated with `trafo_modification_prob > 0`, and the notebook was saved later in the buggy state.

### 3. Dataset Verification Confirms the Bug
- **V2.4 dataset** (`training_networks_saved/ieee30_large/`): 500 files, unique bus counts `[30, 31, 32, 33]`, file sizes vary (173892–187509 bytes)
- **V2.5.2 dataset** (`training_networks_saved/ieee30_large2/`): 500 files, unique bus count `[30]`, file size uniform (173080 bytes)

### 4. Unused `random_topology_for_base` Call
- **Location**: `generate_training_data_with_topology`, the `else` branch (non-ieee9 systems)
- **What it does**: Calls `random_topology_for_base(tmp_net, rng, ..., enable_extra_buses=True, max_extra_buses=3)` which computes `gen_buses`, `load_buses`, `additional_lines` — but these are **never passed** to `create_topology_variant`. The function creates its own topology from scratch.
- **Why it matters**: This is dead code. It wastes computation and misleadingly suggests that `random_topology_for_base` drives the topology, but it doesn't.

## Exact Code Structure (V2.5.2, Cell 22, source lines 280–376)

```
280: [4sp]  # ── 2c) Perturb transformer parameters ──────
281: [4sp]  if trafo_modification_prob > 0.0 and len(net.transformers) > 0:
282: [8sp]      n_perturbed = 0
...
295: [8sp]      if n_perturbed > 0:
296: [12sp]         logger.info(...)
297:        (blank)
298: [4sp]  # ── 3) Optionally add extra buses + lines ────  ← comment at correct level
299: [8sp]      if extra_bus_prob > 0.0 and max_extra_buses > 0:  ← BUG: nested inside 2c's if
...
373: [16sp]                logger.info(...)
374:        (blank)
376: [4sp]  # ── 4) Snapshots ─────────────────────────────  ← back to correct level
```

## The Fix

Dedent lines 299–373 by 4 spaces so `if extra_bus_prob > 0.0` sits at the function body level (4 spaces), not inside the trafo perturbation block (8 spaces).

**Before** (buggy):
```python
    if trafo_modification_prob > 0.0 and len(net.transformers) > 0:
        ...
    # ── 3) Optionally add extra buses + lines ─────────────────────
        if extra_bus_prob > 0.0 and max_extra_buses > 0:   # 8 spaces — WRONG
            ...
```

**After** (fixed):
```python
    if trafo_modification_prob > 0.0 and len(net.transformers) > 0:
        ...
    # ── 3) Optionally add extra buses + lines ─────────────────────
    if extra_bus_prob > 0.0 and max_extra_buses > 0:       # 4 spaces — CORRECT
        ...
```

## Optional Cleanup: Dead Code

In `generate_training_data_with_topology`, the non-ieee9 branch at step 1:
```python
else:
    tmp_net = load_system_from_csv(base_name=base_system, load_dir="grid_model_files")
    gen_buses, load_buses, additional_lines = random_topology_for_base(
        tmp_net, rng, min_gens=1, max_gens=5, min_loads=2,
        enable_extra_buses=True, max_extra_buses=3)
```
The results are never used. Consider removing the `random_topology_for_base` call and `load_system_from_csv` from this branch, since `create_topology_variant` loads the CSV itself.

## Key Files
| File | Purpose | Relevance |
|------|---------|-----------|
| `GNN_Powerflow_V2.5.2.ipynb` Cell 22 | `create_csv_based_topology_variant` definition | Contains the indentation bug |
| `GNN_Powerflow_V2.5.2.ipynb` Cell 20 | `generate_training_data_with_topology` definition | Contains dead code calling `random_topology_for_base` |
| `GNN_Powerflow_V2.5.2.ipynb` Cell 85 | Call site for `networks_ieee30_large_singel_slack2` | Triggers the bug (extra_bus_prob=0.5 but trafo_modification_prob=0.0) |

## Recommendations
1. **Fix the indentation**: Dedent lines 299–373 of `create_csv_based_topology_variant` by 4 spaces
2. **Optional**: Remove the dead `random_topology_for_base` call in `generate_training_data_with_topology`
3. **Regenerate**: Re-run the IEEE30 dataset generation after fixing
