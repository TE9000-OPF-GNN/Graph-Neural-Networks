# Research: Edge Δθ as System-Size-Independent Data Quality Constraint

## Goal
Investigate replacing the current absolute-angle sanity check (`max_angle=180°`, effectively disabled) with an **edge delta_theta constraint** that is physics-based, system-size-independent, and catches non-physical PF solutions.

**Predecessor**: `.docs/research/2026-06-01-delta-theta-scatter-range-and-plot-structure.md` — confirmed that outlier bus angles (+62° in an otherwise [-14°, 0°] system) produce -54° edge deltas that pollute plots. Root cause: `sanity_check_power_flow` has no effective angle filter.

---

## Motivation: Why Absolute Angle is a Poor Metric

| Metric | System-size dependent? | Physics-based bound? | Current default |
|--------|----------------------|---------------------|-----------------|
| |θ_bus| (absolute) | **YES** — 9-bus: max ~20°; 118-bus: max ~50° | No universal bound | `max_angle=180°` (disabled) |
| |Δθ_edge| (per-line) | **NO** — same physics per line regardless of system | Yes: thermal/stability | Not checked |

**Problem with absolute angle**: A 118-bus system may legitimately have bus angles of -50° at the far end of a long radial path. But every INDIVIDUAL line in that path carries only 5–10° of angle difference. The -50° cumulative angle is physically valid; checking it is meaningless.

**Advantage of edge Δθ**: The angle across a single transmission line has physics-based bounds:
- **Steady-state thermal limit**: lines typically operate with |Δθ| < 15–25° 
- **Static stability limit**: |Δθ| < 30° is a common planning constraint (P ∝ sin(Δθ); max power transfer at 90°, practical limit ~30° for security margin)
- **Transient stability**: 90° theoretical max (sin peaks), but no operating system should be near this

A |Δθ| > 30° on ANY single line indicates either:
1. A near-collapse operating point (unrealistic training data)
2. A non-converged/non-physical PF solution (garbage data — our case)

---

## Questions to Investigate

### Q1: What threshold for max_edge_delta_theta?

Candidate thresholds:
| Threshold | Rationale | Expected rejection rate |
|-----------|-----------|----------------------|
| 45° | Very conservative — only catches clearly broken solutions | ~1% |
| 30° | Standard N-1 planning limit — realistic operating envelope | ~3–5%? |
| 20° | Tight — ensures all data is within normal operation | ~10%? |
| 15° | Very tight — textbook "healthy" operation only | ~20%? |

**Need to determine**: What fraction of existing training data exceeds each threshold? This tells us whether the threshold is feasible without massive data loss.

### Q2: Where should the check live?

Options:
1. **In `sanity_check_power_flow`** (DataGen cell 16) — natural place, runs after every PF solve. Reject entire network/snapshot if any edge violates.
2. **In `generate_training_data_with_topology`** (DataGen cell 19) — per-snapshot rejection. Could reject individual snapshots while keeping the network.
3. **In `PowerFlowDataset._create_graph_data`** (Training/Analysis cell 10) — filter at dataset build time. Doesn't prevent saving bad networks but prevents them entering training.
4. **In `_collect_test_set_predictions`** (Analysis cell 23) — eval-time filter only. Quick fix for plots but doesn't fix training data.

### Q3: How to compute edge Δθ efficiently in the sanity check?

The sanity check currently works with PyPSA network objects (not PyG Data objects). Need to compute Δθ from:
```python
v_ang = network.buses_t.v_ang  # [n_snapshots, n_buses] in radians
# For each line/transformer: Δθ = θ_bus0 - θ_bus1
```

### Q4: Should we also check transformers or only lines?

Transformers can have larger angle differences due to tap-ratio effects, but in our flat-pu system (tap_ratio=1.0), transformers behave identically to lines. Include both.

### Q5: What about the existing saved network data?

Options:
- Regenerate all data with the new constraint
- Filter at dataset-build time (cheaper, no regeneration needed)
- Both: filter now for immediate relief, add to DataGen for future generations

---

## Current Implementation (to be extended)

**Location**: `GNN_Powerflow_V2.6_DataGen.ipynb`, cell 16, `sanity_check_power_flow`

```python
def sanity_check_power_flow(
    network: pypsa.Network,
    base_system: str,
    ...
    max_angle: float = 180,  # ← absolute angle, effectively disabled
    ...
) -> None:
```

**Check 2b** (L202–211): Only checks absolute `|θ| > max_angle` — with default 180° this never triggers.

**Call sites** (cells 17, 19, 56, 65): None pass `max_angle` explicitly → all use 180° default.

---

## Proposed Check (sketch)

```python
# New parameter:
max_edge_delta_deg: float = 30.0  # per-line angle difference limit

# New check (after existing check 2b):
# 2c) Edge delta-theta sanity — catches non-physical PF solutions
if max_edge_delta_deg is not None and hasattr(network.buses_t, "v_ang"):
    v_ang = network.buses_t.v_ang  # DataFrame [snapshots × buses]
    for _, branch in pd.concat([network.lines, network.transformers]).iterrows():
        if branch["bus0"] in v_ang.columns and branch["bus1"] in v_ang.columns:
            delta_rad = v_ang[branch["bus0"]] - v_ang[branch["bus1"]]
            delta_deg = np.degrees(delta_rad.values)
            max_dt = np.abs(delta_deg).max()
            if max_dt > max_edge_delta_deg:
                raise RuntimeError(
                    f"{base_system}: edge Δθ={max_dt:.1f}° > {max_edge_delta_deg}° "
                    f"on branch {branch.name} ({branch['bus0']}→{branch['bus1']})"
                )
```

**Performance note**: For a 118-bus system with 186 branches × 24 snapshots, this is ~4500 comparisons — negligible vs PF solve time.

---

## Diagnostic Needed

Run on all available training data to determine what fraction of networks/snapshots would be rejected at each threshold:

```python
# Run over all loaded networks
import pandas as pd

thresholds = [15, 20, 25, 30, 45]
results = {t: 0 for t in thresholds}
total_nets = 0

for net_list_name in ['networks_ieee9_...', 'networks_cigre14_...', 'networks_ieee30_...']:
    nets = globals().get(net_list_name, [])
    for net in nets:
        total_nets += 1
        v_ang_deg = np.degrees(net.buses_t.v_ang.values)
        max_dt = 0
        for _, br in pd.concat([net.lines, net.transformers]).iterrows():
            if br["bus0"] in net.buses.index and br["bus1"] in net.buses.index:
                i0 = list(net.buses.index).index(br["bus0"])
                i1 = list(net.buses.index).index(br["bus1"])
                dt = np.abs(v_ang_deg[:, i0] - v_ang_deg[:, i1]).max()
                max_dt = max(max_dt, dt)
        for t in thresholds:
            if max_dt > t:
                results[t] += 1

print(f"Total networks: {total_nets}")
for t in thresholds:
    print(f"  |Δθ| > {t}°: {results[t]} ({100*results[t]/total_nets:.1f}%)")
```

---

## Recommendations (preliminary — pending diagnostic)

1. **Add `max_edge_delta_deg=30.0` to `sanity_check_power_flow`** — catches non-physical solutions without rejecting valid stressed operation
2. **Implement as check "2d" in the existing function** — minimal disruption
3. **Also add a filter in `_collect_test_set_predictions`** (Analysis) as immediate relief for plotting — skip networks with outlier Δθ at eval time
4. **Run threshold diagnostic** on all training data before committing to a specific value

---

## Open Questions

- [ ] What fraction of existing training data fails each threshold? (diagnostic above)
- [ ] Should rejection be per-snapshot or per-network? (per-snapshot is more granular but complicates the dataset)
- [ ] Literature: what do grid codes specify as max steady-state angle across a line?
- [ ] Should we log warnings vs hard-reject? (currently sanity checks raise RuntimeError → entire network discarded)
