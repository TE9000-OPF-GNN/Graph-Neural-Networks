---
type: BUGFIX
status: done
priority: High
effort: 30m
labels: [training, physics-loss, y-matrix]
depends-on: []
created: 2026-05-21
completed: 2026-05-21
summary: "Fixed get_pypsa_Y_numpy buses_o->buses_i reorder; MSE_fixed now ≤6e-14 for all networks (was up to 269k)"
---

# Fix `get_pypsa_Y_numpy` Silent Bus Mis-Ordering

## Problem

`get_pypsa_Y_numpy` (Cell 8, both training notebooks) contains a guard condition
whose type mismatch means it **always** falls into the wrong branch, returning the
PyPSA subnetwork Y-matrix in `buses_o` order instead of the correct `buses_i`
(network.buses.index) order:

```python
sn_bus_order  = list(sn.buses_i())           # list of bus names / strings
net_bus_order = list(range(len(net_copy.buses)))  # list of ints 0..N-1
if sorted(sn_bus_order) == net_bus_order:    # ALWAYS False — str vs int
    return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]
else:
    return Y_dense   # ← returned in buses_o order for every network
```

For any network where `buses_o ≠ buses_i` order (confirmed for ieee9 non-trivial
topologies, cigre14, ieee30), the returned Y is row/column-permuted. If
`y_matrix_source='auto'` or `'pypsa'` is selected, the physics loss evaluates
`V · (Y_wrong · V*)` — using rows assigned to the wrong buses — producing a massive
irreducible physics floor (measured: ieee9 → 5.6, cigre14 → 0.9, ieee30 → **269,500**
on mean KCL residual with ground-truth voltages).

**Current training is protected** because all S1–S9 sweep cells default to
`y_matrix_sources=("manual",)`. The bug is latent but one parameter change away from
invalidating all physics-loss diagnostics and any future sweep that includes
`y_matrix_source='pypsa'` or `'auto'`.

Priority is **High** because:
- The bug is silent (no error, wrong numbers)
- Experimental magnitude is enormous for ieee30 (269k vs ≤1e-14 for correct Y)
- A future developer adding a `'pypsa'` sweep would see nonsense results with no
  obvious explanation

## Solution

Replace the broken type-mismatched guard with a correct `buses_i == buses_o`
comparison, and apply a proper permutation when they differ.

### Why this exact fix?

- `sn.buses_i()` returns buses in the subnetwork's internal solve order
- `sn.buses_o` is the raw attribute, also subnetwork-internal
- When they match, no reordering needed — return `Y_dense` directly
- When they differ, build a positional map from `buses_o` index → `buses_i` index
  and apply `np.ix_(perm, perm)` to reorder rows and columns simultaneously
- `network.buses.index` order = `buses_i` order (confirmed in `pypsa_bus_ordering.md`)

### Why `buses_i == buses_o` rather than comparing to `network.buses`?

Both `buses_i` and `buses_o` are subnetwork properties. The final reordering maps
`buses_o` → `buses_i`. `network.buses.index` order equals `buses_i` order by
PyPSA convention for the main subnetwork, so after the permutation the returned Y
is already in `network.buses.index` order — which is what `compute_power_flow_residual_from_pred`
and the physics loss expect.

## Scope

**Included:**
- Fix `get_pypsa_Y_numpy` in `GNN_Powerflow_V2.6.1_Training.ipynb` Cell 8
- Fix `get_pypsa_Y_numpy` in `GNN_Powerflow_V2.6_Training.ipynb` Cell 8
- Add a short inline comment explaining the `buses_o → buses_i` permutation

**Not Included:**
- Adding missing `ShuntImpedance` components to `compute_admittance_matrix`
  (separate concern — no shunts in active mixed_1500_wide dataset)
- `use_q_partial_mode` changes (separate feature)
- Sweeping `y_matrix_source='pypsa'` or `'auto'` (user decision after fix is in)
- Analysis notebook Y-matrix references (analysis does not call this function)

## Affected Files

| File | Change |
|------|--------|
| `GNN_Powerflow_V2.6.1_Training.ipynb` | Fix `get_pypsa_Y_numpy` in Cell 8 (~5 lines) |
| `GNN_Powerflow_V2.6_Training.ipynb` | Same fix in the equivalent Cell 8 |

## Implementation Steps

### 1. Locate `get_pypsa_Y_numpy` in both notebooks

Use surgical disk patching (Python patch script). Both notebooks have the function
in their Cell 8 (Y-matrix utilities section).

### 2. Replace the broken guard with the correct permutation logic

**Old code** (remove):
```python
sn_bus_order  = list(sn.buses_i())
net_bus_order = list(range(len(net_copy.buses)))
if sorted(sn_bus_order) == net_bus_order:
    return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]
else:
    return Y_dense
```

**New code** (replace with):
```python
# Reorder from buses_o (subnetwork solve order) → buses_i (network.buses.index order)
sn_buses_i = list(sn.buses_i())
sn_buses_o = list(sn.buses_o)
if sn_buses_i == sn_buses_o:
    return Y_dense
pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
perm = [pos_in_o[b] for b in sn_buses_i]
return Y_dense[np.ix_(perm, perm)]
```

Apply to **both** `GNN_Powerflow_V2.6.1_Training.ipynb` and
`GNN_Powerflow_V2.6_Training.ipynb`.

### 3. Verify with `_diag_perfect_pred.py`

The diagnostic script already measures KCL residual under manual vs PyPSA Y with
ground-truth voltages. After the fix, the `MSE_buggy (PyPSA)` column should drop to
the same order as `MSE_manual` (≤ 1e-14).

Run:
```powershell
python _diag_perfect_pred.py
```

Confirm the `MSE_fixed` column is now ≤ 1e-13 for all networks tested (ieee9, cigre14,
ieee30 at minimum).

### 4. Commit

```powershell
git add -A
git commit -m "fix: get_pypsa_Y_numpy buses_o→buses_i reorder (was always returning wrong order)"
```

## Acceptance Criteria

- [x] `get_pypsa_Y_numpy` no longer contains `list(range(len(net_copy.buses)))` or
  the `sorted(sn_bus_order) == net_bus_order` guard
- [x] Running `_diag_perfect_pred.py` shows KCL residual for PyPSA Y ≤ 1e-13 for
  ieee9, cigre14, and ieee30 with ground-truth voltages (was 5.6, 0.9, 269,500)
- [x] Fix applied to both `GNN_Powerflow_V2.6.1_Training.ipynb` and
  `GNN_Powerflow_V2.6_Training.ipynb`
- [x] All S1–S9 sweep cells continue to produce identical results (they use
  `y_matrix_source='manual'`, which does not call this function)
- [x] Inline comment on the permutation block explains `buses_o → buses_i` in one line
