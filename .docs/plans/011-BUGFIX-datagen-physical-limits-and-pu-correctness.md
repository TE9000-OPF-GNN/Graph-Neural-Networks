---
type: BUGFIX
status: todo
priority: High
effort: 1h
labels: [datagen, pypsa, data-quality]
depends-on: []
created: 2026-05-21
completed:
summary: ""
---

# Fix DataGen Physical Limits and pu Correctness

## Problem

`load_system_from_csv` passes raw `b` values from CSV directly to PyPSA without clamping. Three lines in the local `ieee30_lines.csv` have negative `b` (cross-voltage-level artifact from a stale file) — values like −14.3 pu are 100–1000× the normal line charging range and corrupt the Y-bus and PF results for IEEE30. Additionally, `n.sn_mva` defaults to `1.0` instead of `100.0`, causing diagnostic outputs and MW/MVAr conversions to show wrong magnitude. `sanity_check_power_flow` has no KCL consistency check, so a network with modeling inconsistencies (e.g. the negative-b case) can pass all existing checks and produce silently wrong results.

**Note (2026-05-21)**: Task F (revert `q_min_pu`/`q_max_pu`) was already completed — verified no such attributes exist in the current notebook.

## Solution

Three targeted changes, all using the surgical disk patching protocol:

**A. Clamp `b` to `≥ 0` in `load_system_from_csv`**  
Wrap the `b_val` assignment in `max(..., 0.0)`. The fix is defense-in-depth: OneDrive CSVs already have `b ≥ 0` after the cross-VL fix, but stale local files do not. One line change; no side effects.

**B. Set `n.sn_mva = 100.0` immediately after `pypsa.Network()`**  
`sn_mva` is metadata only — PyPSA `pf.py` does not read it at any point (zero hits in source). Safe to add without affecting the PF solution. Makes `n.summary()` and any downstream MW/MVAr diagnostic show the correct base.

**G. Add KCL balance check to `sanity_check_power_flow`**  
After all existing checks pass, compute `|gen_P − load_P − losses| / |gen_P|`. This is expected to be `< 1e-5` for a well-converged AC PF. Values `> 1e-3` indicate a modeling inconsistency that the solver "solved" by converging to a wrong operating point. New `max_balance_err=1e-3` parameter (default keeps backward compatibility).

**Why not tasks C and D?**  
Research confirmed: changing `s_nom=100` (task C) would make PyPSA compute `r_pu = r/100`, shrinking transformer admittances 100×. Changing `v_nom=actual_kV` (task D) would make PyPSA compute `r_pu = r/kV²`, shrinking line admittances 17,424× for 132 kV buses. Both would corrupt the Y-bus. The current flat-pu system (`v_nom=1.0`, `s_nom=1.0`) is mathematically self-consistent. These are **not** to be implemented.

## Scope

**Included:**
- `load_system_from_csv` in cell 19 of `GNN_Powerflow_V2.6_DataGen.ipynb`: clamp b (task A) + set sn_mva (task B)
- `sanity_check_power_flow` in cell 16: add `max_balance_err` parameter + check 2c (task G)

**Not Included:**
- Transformer `s_nom` or `v_nom`/`tap_ratio` changes (confirmed dangerous)
- Retroactive regeneration of existing datasets
- Q-limit enforcement (would require PV→PQ bus-switching loop; out of scope)
- `create_csv_based_topology_variant` perturbation path (already clamps `b ≥ 0` post-perturbation)

## Affected Files

| File | Change |
|------|--------|
| `GNN_Powerflow_V2.6_DataGen.ipynb` cell 19 (`load_system_from_csv`) | Clamp `b_val ≥ 0`; add `n.sn_mva = 100.0` |
| `GNN_Powerflow_V2.6_DataGen.ipynb` cell 16 (`sanity_check_power_flow`) | Add `max_balance_err=1e-3` param; add check 2c |

## Implementation Steps

### 1. Clamp b in `load_system_from_csv` (cell 19)

**Before** (line 36):
```python
        b_val = float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0)
```

**After**:
```python
        b_val = max(float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0), 0.0)
```

### 2. Set `sn_mva` in `load_system_from_csv` (cell 19)

**Before** (line 16–17):
```python
    n = pypsa.Network()

    # Buses
```

**After**:
```python
    n = pypsa.Network()
    n.sn_mva = 100.0  # cosmetic: all CSV data is on 100 MVA base

    # Buses
```

### 3. Add balance check to `sanity_check_power_flow` (cell 16)

**Before** — function signature (line 163–171):
```python
def sanity_check_power_flow(
    network: pypsa.Network,
    base_system: str,
    require_connected: bool = True,
    min_loads: int = 1,
    voltage_max_threshold: float = 1.5,
    voltage_min_threshold: float = 0.5,
    max_angle: float = 180
) -> None:
```

**After**:
```python
def sanity_check_power_flow(
    network: pypsa.Network,
    base_system: str,
    require_connected: bool = True,
    min_loads: int = 1,
    voltage_max_threshold: float = 1.5,
    voltage_min_threshold: float = 0.5,
    max_angle: float = 180,
    max_balance_err: float = 1e-3,
) -> None:
```

Then, insert check 2c **before** the final `logger.info(...)` line (currently line 223).

**Before** — end of function (lines 222–227):
```python

    logger.info(
        f"{base_system}: PF sanity OK "
        f"(buses={len(network.buses)}, gens={len(network.generators)}, "
        f"loads={len(network.loads)})"
    )
```

**After**:
```python

    # 2c) System power balance (KCL)
    if hasattr(network, "generators_t") and hasattr(network.generators_t, "p"):
        gen_p   = float(network.generators_t.p.values.sum())
        load_p  = float(network.loads_t.p.values.sum())
        line_loss = float((network.lines_t.p0.values + network.lines_t.p1.values).sum())
        traf_arr = network.transformers_t.p0.values + network.transformers_t.p1.values
        traf_loss = float(traf_arr.sum()) if traf_arr.size > 0 else 0.0
        balance_err = abs(gen_p - load_p - line_loss - traf_loss) / max(abs(gen_p), 1e-6)
        if balance_err > max_balance_err:
            raise RuntimeError(
                f"{base_system}: power balance error {balance_err:.2e} > {max_balance_err:.0e} "
                f"(gen={gen_p:.4f}, load={load_p:.4f}, losses={line_loss+traf_loss:.6f})"
            )

    logger.info(
        f"{base_system}: PF sanity OK "
        f"(buses={len(network.buses)}, gens={len(network.generators)}, "
        f"loads={len(network.loads)})"
    )
```

## Acceptance Criteria

- [ ] `python -c "import json; src=''.join(json.load(open('GNN_Powerflow_V2.6_DataGen.ipynb'))['cells'][19]['source']); assert 'b_val = max(' in src"` exits 0
- [ ] `python -c "import json; src=''.join(json.load(open('GNN_Powerflow_V2.6_DataGen.ipynb'))['cells'][19]['source']); assert 'n.sn_mva = 100.0' in src"` exits 0
- [ ] `python -c "import json; src=''.join(json.load(open('GNN_Powerflow_V2.6_DataGen.ipynb'))['cells'][16]['source']); assert 'max_balance_err' in src"` exits 0
- [ ] After re-running cells 16 and 19: `load_system_from_csv("ieee9")` returns a network with `n.sn_mva == 100.0` and `n.transformers["s_nom"].iloc[0] ≈ 1.0` (sn_mva did not change transformer s_nom)
- [ ] `sanity_check_power_flow(n_ieee9, "ieee9")` completes without RuntimeError; PyPSA log has no power balance warning
- [ ] `sanity_check_power_flow(n_ieee30, "ieee30")` completes without RuntimeError (b clamp prevents negative-b crash)
- [ ] No `q_min_pu` or `q_max_pu` strings anywhere in the notebook (task F remains clean)
