# Research: CIGRE14 Physics-Enabled Training Regression

**Date:** 2026-04-10  
**Notebook:** `GNN_Powerflow_V2.5.2.ipynb`  
**Symptom:** Physics-enabled models appear to regress (perform worse) on 14-bus CIGRE system, similar to the previously-fixed 9-bus IEEE bus ordering issue.

---

## Executive Summary

The physics-enabled training regression reported for CIGRE14 is **primarily a display/evaluation artifact**, not a training bug. The root cause is `reconstruct_bus_quantities_from_output` using `network.buses['type']` which is **empty** for CSV-based networks. This causes per-network evaluation plots to misidentify all buses as Slack, corrupting the displayed metrics and making physics models appear to perform worse.

The training pipeline itself (physics loss, Y-matrix, masking) is correctly implemented and is NOT affected by this bug.

---

## Findings

### FINDING 1: `reconstruct_bus_quantities_from_output` — EVALUATION BUG (HIGH)

**Location:** Cell 40, line 192  
**Impact:** Per-network comparison scatter plots and their metrics  
**Scope:** Evaluation/visualization only — does NOT affect training

```python
# BUG: network.buses['type'] is "" for all CSV-based networks
bus_types = network.buses['type'].values

for i, bus_type in enumerate(bus_types):
    if bus_type == 'PQ':     # Never matches (type is "")
        ...
    elif bus_type == 'PV':   # Never matches (type is "")
        ...
    else:  # Slack            # ALL buses fall here
        v_pred[i]   = v_true_mag[i]   # ← Vmag from truth (wrong for PQ)
        ang_pred[i] = v_true_ang[i]   # ← Vang from truth (wrong for PQ/PV)
        p_pred[i]   = node_pred[i, 2] # ← P from prediction (wrong for PQ/PV)
        q_pred[i]   = node_pred[i, 3] # ← Q from prediction (wrong for PQ)
```

**Effect on displayed metrics:**
- Vmag/Vang errors appear artificially LOW (truth is used for all buses)
- P/Q errors appear artificially HIGH (predictions used where known values should be)
- This makes ALL models look worse on P/Q, but physics models may show more distortion since they train P differently

**Fix:** Use generator `control` field instead of `buses['type']`, consistent with `_create_graph_data` and `predict_network_results_with_masks`:

```python
# FIXED approach:
gen_bus_ctrl = {}
for _, gen in network.generators.iterrows():
    gen_bus_ctrl[gen["bus"]] = gen["control"]

for i, bus in enumerate(network.buses.index):
    bus_type = gen_bus_ctrl.get(bus, "PQ")
    ...
```

### FINDING 2: `load_system_from_csv` — MISSING `buses['type']` (ROOT CAUSE)

**Location:** Cell 22, lines 20-26  
**Impact:** All CSV-based networks (CIGRE14, IEEE30, IEEE39, IEEE57, IEEE118, CIGRE14DER)

```python
# Current: buses['type'] defaults to "" (empty string)
n.add("Bus", bus_idx,
    v_nom=1.0,
    v_mag_pu_set=vmag_pu,
    v_mag_pu_min=0.9,
    v_mag_pu_max=1.1)
# Missing: type= parameter
```

**Contrast with 9-bus `add_buses` (Cell 21, line 100):**
```python
# 9-bus explicitly sets type:
network.add("Bus", f"Bus {i}", ..., type=bus_type)
```

**Fix option A:** Set `type` on buses after generators are loaded:
```python
# After generator loop in load_system_from_csv:
for gen_id, row in gens_df.iterrows():
    bus_name = row.get("bus_id")
    ctrl = ...  # Slack/PV/PQ as already computed
    n.buses.loc[bus_name, "type"] = ctrl
# Mark remaining buses as PQ
n.buses["type"] = n.buses["type"].replace("", "PQ")
```

**Fix option B:** Fix `reconstruct_bus_quantities_from_output` directly (Finding 1), which is the real consumer.

### FINDING 3: Training Pipeline — CORRECTLY IMPLEMENTED (No bug)

Verified functions that correctly use generator `control` / graph data masks:

| Function | Bus type source | Correct? |
|---|---|---|
| `_create_graph_data` (Cell 32) | `gen["control"]` → masks | ✅ |
| `_masked_mse_loss` (Cell 37) | `batch.slack/pv/pq_mask` | ✅ |
| `physics_informed_loss_batch` (Cell 34) | `batch.slack/pv/pq_mask` | ✅ |
| `compute_power_flow_residual_from_pred` (Cell 34) | `bus_masks` argument | ✅ |
| `evaluate_gnn_on_test_set` (Cell 40) | `data.slack/pv/pq_mask` | ✅ |
| `predict_network_results_with_masks` (Cell 40) | `gen["control"]` | ✅ |

**The STSI260404 bugfix in `_create_graph_data`** (`.loc[snapshot, buses]` for column alignment) is also verified to be in place and working correctly.

### FINDING 4: Y-Matrix — CORRECT for Default Settings

- Default `y_matrix_source` in `run_hparam_sweep` = `"manual"` ✅
- `compute_admittance_matrix` uses `bus_to_idx = {b: i for i, b in enumerate(buses)}` where `buses = list(network.buses.index)` — consistent with graph data ordering ✅
- Y-matrix dimensions align with node ordering in batched graphs ✅

### FINDING 5: `get_pypsa_Y_numpy` — LATENT BUG (LOW PRIORITY)

**Location:** Cell 30, function `get_pypsa_Y_numpy`

```python
sn_bus_order  = list(sn.buses_i())           # Strings: ["B0", "B1", ...]
net_bus_order = list(range(len(net_copy.buses)))  # Ints: [0, 1, 2, ...]

if sorted(sn_bus_order) == net_bus_order:  # ALWAYS False for string bus names
    return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]
else:
    return Y_dense  # Raw PyPSA ordering — may not match network.buses.index
```

**Impact:** Only matters when `y_matrix_source="pypsa"` or `"auto"` mode selects PyPSA Y. Default `"manual"` mode is not affected. If activated, the `"auto"` mode would detect a mismatch between manual and PyPSA Y (due to ordering), and then incorrectly use the mis-ordered PyPSA Y.

### FINDING 6: Display-Only `buses['type']` References

Two hardcoded references in Cell 45 (visualization) also use `buses['type']` directly:
- Line 366: `bus_labels = [f"Bus {bus}\n({network.buses.loc[bus, 'type']})" ...]`
- Line 411: `bus_labels = [f"Bus {bus}\n({network.buses.loc[bus, 'type']})" ...]`

These show empty labels `()` for CSV networks. Should use `get_bus_type_label(network, bus)` which already correctly derives type from generator `control`.

---

## Recommended Fix Priority

1. **Fix `reconstruct_bus_quantities_from_output`** — highest impact, fixes the evaluation artifact that makes physics models appear to regress
2. **Fix `load_system_from_csv` to set `buses['type']`** — fixes root cause for all downstream code
3. **Fix Cell 45 hardcoded `buses['type']` label references** — cosmetic but consistent
4. **Fix `get_pypsa_Y_numpy` string comparison** — latent bug, low priority since default uses manual Y

---

## Confidence Assessment (FAR Scale)

| Finding | Confidence | Evidence |
|---|---|---|
| `reconstruct_bus_quantities_from_output` bug | **F** (Fact) | Code reads `buses['type']` which is verified empty for CSV networks |
| `load_system_from_csv` missing type | **F** (Fact) | No `type=` parameter in `n.add("Bus", ...)` call |
| Training pipeline correct | **F** (Fact) | All 6 training functions verified to use generator `control` / masks |
| Physics regression is display artifact | **A** (Assessment) | Need to verify by checking actual test_metrics numbers (from `evaluate_gnn_on_test_set`) vs scatter plot metrics (from `evaluate_model`) |
| `get_pypsa_Y_numpy` latent bug | **F** (Fact) | String vs int comparison always fails |

---

## Open Questions

1. **Are `test_metrics` numbers (from `evaluate_gnn_on_test_set`) actually worse for physics models?** If they are, there's a genuine training issue beyond the display bug. If they're fine, it confirms the display artifact hypothesis.
2. **What `w_phys` / physics hyperparameters work best for CIGRE14?** The same hyperparameters that work for IEEE9 may not be optimal for the larger, different-topology CIGRE14 system.
3. **Does the `b = b_val / 2` in `load_system_from_csv` matter?** The shunt susceptance is stored as `(b1+b2)/2` — both PyPSA and manual Y use the same value, so the physics residual should still be zero for correct solutions. Not a cause of regression.
