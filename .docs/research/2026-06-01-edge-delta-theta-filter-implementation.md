# Research: Edge Δθ as Data Quality Filter — Implementation Design

## Goal
Design the implementation of `max_edge_delta_deg` in `sanity_check_power_flow` to reject non-physical PF solutions. This extends the earlier research (`.docs/research/2026-06-01-delta-theta-sanity-constraint.md`) with verified code structure findings.

---

## Key Findings

### 1. Current Check is Ineffective (check 2b, L202–211)

- **Location**: `GNN_Powerflow_V2.6_DataGen.ipynb`, cell 16, `sanity_check_power_flow`
- **Signature**: `max_angle: float = 180` — effectively disabled
- **What it checks**: `|θ_bus| > 180°` (absolute angle of any bus)
- **Call sites** (cells 17, 19, 56, 65): None pass `max_angle` → all use 180° default
- **Why it fails**: An individual bus at +62° passes the 180° check trivially. The metric is also system-size-dependent (a 118-bus chain legitimately reaches cumulative -50°).

### 2. Network Structure: One Snapshot Per Network Object

- **Location**: `GNN_Powerflow_V2.6_DataGen.ipynb`, cell 18 (generation loop)
- Each iteration of `while len(networks) < num_scenarios` produces one `pypsa.Network` with **one** solved snapshot
- `sanity_check_power_flow(net, ...)` is called inside the `try:` block (via `generate_training_data_with_topology` in cell 19, L456)
- If it raises `RuntimeError`, the `except Exception` at L345 catches it, increments `failure_reasons["other"]`, and the network is discarded
- **Implication**: Per-network rejection is the existing granularity — no need for per-snapshot logic

### 3. Branch Topology Available at Check Time

At `sanity_check_power_flow` execution, the network object has:
- `network.lines` DataFrame with `bus0`, `bus1` columns (string bus names)
- `network.transformers` DataFrame with `bus0`, `bus1` columns
- `network.buses_t.v_ang` DataFrame with columns = bus names, single row (one snapshot)
- `network.buses.index` = all bus names

The connectivity graph is already built in the function (L183–187) using `networkx` for connectivity check. Bus name → column lookup is direct.

### 4. Training Dataset Computes Same Metric (Validation)

- **Location**: `GNN_Powerflow_V2.6.2_Training.ipynb`, cell 11, L322–324
- ```python
  theta_true = y[:, 1]  # absolute angles from PF
  y_delta_theta = theta_true[edge_index_fwd[0]] - theta_true[edge_index_fwd[1]]
  ```
- This uses integer indices (`bus_to_i` mapping), but the physics is identical to what we'll compute in the sanity check using bus names.

### 5. Confirmed Bad Data Example

From scatter diagnosis research:
| Network | Bus angle range | Max |Δθ_edge| | Status |
|---------|----------------|-----------------|--------|
| 0 (ieee57) | [-23.8°, 0.0°] | 12.8° | Clean |
| 2 (ieee57) | [-13.8°, **+65.6°**] | **54.3°** | Non-physical |

A threshold of 30° would reject network 2 without touching network 0.

### 6. Radians Bug in Post-Generation Diagnostic Plots (DataGen cell 23)

**Bug**: `plot_solved_state_matrix` (L381) and `plot_solved_state_matrix_stats` (L581) both label the V_ang column as "V_ang [°]" but plot **raw radians** from `net.buses_t.v_ang` without conversion.

- **Location**: `GNN_Powerflow_V2.6_DataGen.ipynb`, cell 23
- `plot_solved_state_matrix`, L422–423:
  ```python
  if not net.buses_t.v_ang.empty:
      va_all.append(net.buses_t.v_ang.values.flatten())  # ← radians!
  ```
- `plot_solved_state_matrix_stats`, L616–617: same pattern, same bug
- **Axis label** (L412, L605): `"V_ang [°]"` — misleading

**Effect**: Values display as ~[-0.25, 0] for a 9-bus system (which is actually [-14°, 0°]). This makes the diagnostic appear as though all angles are "within 1 degree" — masking the true spread. The confirmed bad network with +62° bus angle would show as +1.09 radians on the plot, easily mistaken for a ~1° value.

**Fix**: Add `np.degrees()` conversion when collecting `va_all`:
```python
va_all.append(np.degrees(net.buses_t.v_ang.values.flatten()))
```

**Note**: The per-system histograms that show correct angles come from a *different* function in **cell 22** (`plot_dataset_distributions`), which correctly converts: `v_ang_all.append((net.buses_t.v_ang.values * (180.0 / np.pi)).flatten())`. The two functions in **cell 23** (`plot_solved_state_matrix` at L381 and `plot_solved_state_matrix_stats` at L581) both omit this conversion.

---

## Implementation Plan — Hybrid: Outlier Detection + Hard Cap

### Design Philosophy

A fixed threshold (e.g., 30°) is problematic:
- **Too tight**: Rejects legitimately stressed networks with uniformly high Δθ (e.g., all edges at 20–25° in a heavily-loaded system)
- **Too loose**: Misses subtler outliers (e.g., one edge at 18° when all others are 2°)
- **Physics**: Δθ can theoretically reach 90° (max power transfer); stressed networks legitimately operate at 20–30° on critical corridors

The actual problem: **within-network outliers** — one edge with extreme Δθ while the rest are normal, indicating a non-converged or non-physical PF solution.

### Hybrid Approach: IQR Outlier + Hard Cap

Two independent checks:
1. **IQR outlier detection** (primary): Flag if `max(|Δθ|) > Q3 + k × IQR` within the same network
2. **Hard safety cap** (secondary): Flag if any `|Δθ| > 60°` regardless (no physical line operates here in steady state)

### Location: Cell 16, insert after L211 (after check 2b)

```python
    # 2d) Edge delta-theta outlier detection — catches non-physical PF solutions
    #     Uses IQR-based outlier detection: a single extreme Δθ in an otherwise
    #     well-behaved network indicates bad convergence, not legitimate stress.
    #     Hard cap at 60° catches universally non-physical cases.
    if max_edge_delta_iqr_k is not None and hasattr(network.buses_t, "v_ang"):
        v_ang_rad = network.buses_t.v_ang  # DataFrame [1 × n_buses] for single-snapshot
        branches = pd.concat(
            [network.lines[["bus0", "bus1"]], network.transformers[["bus0", "bus1"]]],
            ignore_index=True
        )
        valid = branches["bus0"].isin(v_ang_rad.columns) & branches["bus1"].isin(v_ang_rad.columns)
        branches = branches[valid]
        if len(branches) >= 3:  # need at least 3 edges for meaningful IQR
            ang0 = v_ang_rad[branches["bus0"].values].values.flatten()
            ang1 = v_ang_rad[branches["bus1"].values].values.flatten()
            delta_deg = np.abs(np.degrees(ang0 - ang1))
            max_dt = delta_deg.max()
            worst_idx = int(delta_deg.argmax())
            worst_branch = f"{branches.iloc[worst_idx]['bus0']}→{branches.iloc[worst_idx]['bus1']}"

            # Check 1: Hard cap — no line ever exceeds 60° in steady-state
            if max_dt > max_edge_delta_abs:
                raise RuntimeError(
                    f"{base_system}: edge |Δθ|={max_dt:.1f}° > {max_edge_delta_abs}° hard cap "
                    f"on branch {worst_branch}"
                )

            # Check 2: IQR outlier — max Δθ is abnormally large vs peers
            q1, q3 = np.percentile(delta_deg, [25, 75])
            iqr = q3 - q1
            upper_fence = q3 + max_edge_delta_iqr_k * iqr
            if iqr > 0.5 and max_dt > upper_fence:  # iqr > 0.5° avoids triggering on near-zero spread
                raise RuntimeError(
                    f"{base_system}: edge |Δθ| outlier: max={max_dt:.1f}° > fence={upper_fence:.1f}° "
                    f"(Q3={q3:.1f}°, IQR={iqr:.1f}°, k={max_edge_delta_iqr_k}) "
                    f"on branch {worst_branch}"
                )
```

### Parameter Addition

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
    max_edge_delta_iqr_k: float = 5.0,   # NEW — IQR multiplier for outlier detection
    max_edge_delta_abs: float = 60.0,     # NEW — hard physics cap (degrees)
) -> None:
```

### Default Values

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_edge_delta_iqr_k` | 5.0 | Conservative: max must be >5× IQR above Q3. For the confirmed bad case: Q3≈5°, IQR≈3°, fence=20°, max=54° → clearly caught. For uniformly stressed: Q3≈20°, IQR≈5°, fence=45° → allows up to 45°. |
| `max_edge_delta_abs` | 60.0 | No single transmission line operates at >60° steady-state (sin(60°)=87% of max transfer — beyond all credible operating envelopes). Safety net only. |

### Why IQR Over Other Methods

| Concern | IQR handles it | Alternatives fail |
|---------|---------------|-------------------|
| Small networks (9-bus, ~12 edges) | IQR is defined for n≥4; we require n≥3 with guard | Z-score needs normality assumption |
| Uniformly stressed systems | High Q3 → high fence → won't reject | Fixed threshold rejects them |
| Near-zero spread (slack-dominated) | `iqr > 0.5°` guard prevents division issues | Ratio-to-median: median≈0 → blows up |
| One extreme outlier | max >> fence by definition | All methods catch this |

### Performance

For 118-bus system with ~186 branches × 1 snapshot:
- Vectorized numpy: ~186 subtractions + 1 max → negligible (<0.1 ms)
- No loop over branches (unlike the proposed sketch in the earlier research doc)

---

## Integration Details

### Call Sites — No Changes Needed

All 4 call sites use keyword args and the new parameters have defaults. They will automatically apply outlier detection without modification:
- Cell 17, L341: ieee9 single-network test → defaults apply
- Cell 19, L456: `generate_training_data_with_topology` main path → defaults apply
- Cell 56, L179: CSV smoke test → defaults apply
- Cell 65, L11/L28: unit test cells → defaults apply

### Failure Counting

When the new check raises `RuntimeError`, it's caught by the generation loop's `except Exception` (cell 18, L345) and counted as `failure_reasons["other"]`. 

**Optional enhancement**: Add a specific failure reason counter. Could change the generation loop to detect edge-delta failures:
```python
except RuntimeError as e:
    if "edge" in str(e) and "Δθ" in str(e):
        failure_reasons["edge_delta_outlier"] = failure_reasons.get("edge_delta_outlier", 0) + 1
    else:
        failure_reasons["other"] += 1
```

### Eval-Time Filter (Optional, Lower Priority)

For existing saved networks that were generated without this filter, add an eval-time skip in `_collect_test_set_predictions` (Analysis cell 23). This is a separate task — not required for the core implementation.

---

## Threshold Diagnostic (to run on actual data)

Validates both the IQR outlier detection (k=5) and hard cap (60°) against existing training data:

```python
import numpy as np, pandas as pd

iqr_k_values = [3, 5, 7, 10]
hard_caps = [30, 45, 60]
results_iqr = {k: 0 for k in iqr_k_values}
results_cap = {c: 0 for c in hard_caps}
total = 0

for net in networks:  # replace with actual list variable
    total += 1
    v_ang = net.buses_t.v_ang  # single-snapshot DataFrame
    branches = pd.concat([net.lines[["bus0","bus1"]], net.transformers[["bus0","bus1"]]], ignore_index=True)
    valid = branches["bus0"].isin(v_ang.columns) & branches["bus1"].isin(v_ang.columns)
    branches = branches[valid]
    if len(branches) < 3:
        continue
    ang0 = v_ang[branches["bus0"].values].values.flatten()
    ang1 = v_ang[branches["bus1"].values].values.flatten()
    delta_deg = np.abs(np.degrees(ang0 - ang1))
    max_dt = delta_deg.max()

    # Hard cap check
    for c in hard_caps:
        if max_dt > c:
            results_cap[c] += 1

    # IQR outlier check
    q1, q3 = np.percentile(delta_deg, [25, 75])
    iqr = q3 - q1
    if iqr > 0.5:
        for k in iqr_k_values:
            if max_dt > q3 + k * iqr:
                results_iqr[k] += 1

print(f"Total networks: {total}")
print("\nHard cap rejections:")
for c in hard_caps:
    print(f"  |Δθ| > {c}°: {results_cap[c]:4d} ({100*results_cap[c]/total:.1f}%)")
print("\nIQR outlier rejections (iqr > 0.5° guard):")
for k in iqr_k_values:
    print(f"  k={k}: {results_iqr[k]:4d} ({100*results_iqr[k]/total:.1f}%)")
```

---

## Constraints & Considerations

1. **Flat-pu system**: All transformers have `tap_ratio=1.0` → they behave as lines for angle difference purposes. Include both lines and transformers.
2. **Single-snapshot networks**: Each network has exactly 1 solved PF → `v_ang_rad` is a 1-row DataFrame. The vectorized implementation handles this naturally.
3. **Bus name matching**: `v_ang.columns` uses the same bus names as `lines["bus0"]`/`lines["bus1"]` (verified in cell 19 L456 context).
4. **No backward compatibility issue**: New parameters default to `iqr_k=5.0, abs=60.0` — existing call sites need no changes but will now reject more networks (slightly lower success rate, compensated by `max_attempts`).
5. **Minimum edge count**: IQR requires ≥3 edges to be meaningful. Guard `len(branches) >= 3` prevents issues on degenerate networks. The hard cap still applies regardless.

---

## Recommendations

1. **Implement hybrid check (IQR + hard cap)** as check 2d in `sanity_check_power_flow` — vectorized, ~15 lines of code
2. **No call-site changes needed** — default values apply everywhere
3. **Run threshold diagnostic** on ieee9/cigre14/ieee30 training data to validate that k=5 doesn't over-reject (expected: catches only the confirmed bad cases)
4. **Optional**: Add `"edge_delta_outlier"` failure reason counter in generation loop for visibility
5. **Separate task**: Eval-time filter for existing saved data (Analysis notebook)
6. **Fix radians bug in diagnostic plots** (DataGen cell 23): `plot_solved_state_matrix` (L423) and `plot_solved_state_matrix_stats` (L617) both plot raw radians with "V_ang [°]" label. Add `np.degrees()` conversion to match the correct implementation in cell 22's `plot_dataset_distributions`. Two-line fix:
   - L423: `va_all.append(np.degrees(net.buses_t.v_ang.values.flatten()))`
   - L617: `va_all.append(np.degrees(net.buses_t.v_ang.values.flatten()))`

---

## Open Questions

- [ ] What % of existing training data fails the IQR(k=5) + 60° cap check? (diagnostic needed — load datasets)
- [ ] Is k=5 too conservative or too aggressive? (need empirical data from training sets)
- [ ] Should `iqr > 0.5°` guard be tunable or is it fine as hardcoded? (probably fine — only matters for near-DC-flat networks)
