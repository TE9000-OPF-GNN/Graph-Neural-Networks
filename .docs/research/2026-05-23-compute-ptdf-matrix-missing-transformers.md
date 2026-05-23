# Bug: `compute_ptdf_matrix` Omits Transformers from B Matrix

**Date**: 2026-05-23
**Severity**: High (incorrect PTDF for any network with transformers)
**Location**: `GNN_Powerflow_V2.6.1_Training.ipynb`, cell 8, `compute_ptdf_matrix(network)`
**Status**: Confirmed, not yet fixed

---

## Summary

`compute_ptdf_matrix` builds the bus susceptance matrix B by looping over `network.lines`
only. **Transformers are completely omitted.** This means:

- B matrix is missing transformer susceptance contributions
- `B_red_inv` is incorrect for any network where buses connect via transformers
- PTDF matrix is wrong (though currently unused since `use_ptdf_loss=False`)
- DC angle computation `θ = B_red_inv @ P` produces garbage for affected buses

---

## Evidence

### Affected networks

| Network | Transformers | Impact |
|---------|-------------|--------|
| **ieee9** | 3 (Bus1↔Bus4, Bus3↔Bus6, Bus8↔Bus2) | **CRITICAL** — gen buses ONLY connected via transformers; B has zero rows for buses 1,2,3 → B_red singular |
| **cigre14** | 2 (B0↔B1, B0↔B12) | High — slack bus B0 only connects via transformers |
| **ieee30** | 4 (4↔12, 6↔9, 6↔10, 28↔27) | High — transformer-connected buses have incomplete B |
| **ieee39** | transformers present | Same issue |
| **ieee57** | transformers present | Same issue |
| **ieee118** | transformers present | Same issue |

### Code (cell 8, lines 20-29)

```python
for i, line in enumerate(lines):
    b_val = 1.0 / network.lines.loc[line, "x"] if network.lines.loc[line, "x"] != 0 else 0.0
    from_idx = bus_to_idx[network.lines.loc[line, "bus0"]]
    to_idx = bus_to_idx[network.lines.loc[line, "bus1"]]
    B[from_idx, from_idx] += b_val
    B[to_idx, to_idx] += b_val
    B[from_idx, to_idx] -= b_val
    B[to_idx, from_idx] -= b_val
    A[i, from_idx] = b_val
    A[i, to_idx] = -b_val
# ← NO transformer loop follows
```

### Consequence for ieee9

```
Transformer 1:  Bus 1 → Bus 4  (x=0.0576)
Transformer 4:  Bus 3 → Bus 6  (x=0.0509)
Transformer 7:  Bus 8 → Bus 2  (x=0.0625)
```

Gen buses 1, 2, 3 are connected to the rest of the network ONLY through transformers.
Without transformer susceptance in B:
- Rows/columns for buses 1, 2, 3 are all-zero in B
- After removing slack (bus 1): B_red has zero rows for buses 2, 3
- `np.linalg.inv(B_red)` fails → falls back to `pinv` → incorrect angles

---

## Why it wasn't caught earlier

- `use_ptdf_loss=False` is the default — the PTDF matrix was never actively used in training
- The PTDF auxiliary loss (when enabled) was learning bilinear weights that could compensate
- `compute_admittance_matrix` (same cell 8) DOES include transformers correctly — used by physics loss
- No unit test validates PTDF matrix against known DC power flow angles

---

## Fix

Add transformer susceptance to B matrix in `compute_ptdf_matrix`. In the flat-pu system
(tap_ratio=1.0), a transformer is electrically a series impedance — contributes `1/x` to B
just like a line.

```python
# After the lines loop, ADD:
for trafo in network.transformers.index:
    row = network.transformers.loc[trafo]
    # Prefer x_pu_eff (system-base) if available; fallback to raw x
    if "x_pu_eff" in row.index and pd.notna(row.get("x_pu_eff")) and float(row["x_pu_eff"]) != 0:
        x_val = float(row["x_pu_eff"])
    else:
        x_val = float(row["x"])
    if x_val == 0:
        continue
    b_val = 1.0 / x_val
    from_idx = bus_to_idx[row["bus0"]]
    to_idx = bus_to_idx[row["bus1"]]
    B[from_idx, from_idx] += b_val
    B[to_idx, to_idx] += b_val
    B[from_idx, to_idx] -= b_val
    B[to_idx, from_idx] -= b_val
    # Note: transformer NOT added to incidence A — PTDF gives LINE flows only.
    # To also predict transformer flows, add rows to A (separate decision).
```

### What about the PTDF incidence matrix A?

Two options:
1. **Lines only** (current): PTDF maps injections → line flows. Transformer flows not predicted.
2. **All branches**: Extend A to include transformer rows. PTDF maps injections → all branch flows.

For V2.6 (PTDF auxiliary loss disabled): option 1 is sufficient — only B needs fixing.
For V2.7 (DC baseline): only `B_red_inv` is needed for angle computation — A is irrelevant.

---

## Validation

After fix, verify:
1. `B_red` is full-rank for ieee9 (no zero rows/columns)
2. DC angles match PyPSA `lpf()` output: `θ_dc ≈ network.buses_t.v_ang` from DC PF
3. PTDF matrix satisfies `PTDF @ ΔP ≈ Δflows` for small injection perturbations

---

## Relationship to V2.7 (feature/ptdf-residual-gnn)

This bug becomes **safety-critical** in V2.7 where `B_red_inv` directly produces the DC
baseline angles fed as GNN input. Incorrect angles → incorrect baseline → model trains on
wrong residuals → poor convergence. The fix is a prerequisite for V2.7 and should be
applied to main first, then cherry-picked or inherited by the feature branch.
