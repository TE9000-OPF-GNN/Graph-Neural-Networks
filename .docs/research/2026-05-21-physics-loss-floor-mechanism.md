# Research: Physics Loss Floor — Causes, Y-Matrix Discrepancy, MSE Floor

**Date**: 2026-05-21  
**Updated**: 2026-05-21 (session 2 — root cause confirmed experimentally)  
**Context**: S1 (fixed `w_phys`) and S2 (adaptive `fraction_physics`) sweeps on `networks_mixed_1500_wide`. Both show a physics loss floor. Original question: why do manual Y and PyPSA Y differ, and is this causing the floor?

---

## Summary of Findings (session 2 — supersedes earlier analysis below)

| Question | Answer |
|---|---|
| Why does the Y-matrix differ? | `get_pypsa_Y_numpy` returns Y in `sn.buses_o` order (wrong) |
| Does this cause the floor in current training? | **No** — sweeps use `y_matrix_source='manual'` (correct Y) |
| Is there a floor with manual Y + perfect predictions? | **No** — MSE < 1e-14 (floating-point noise only, confirmed experimentally) |
| What explains the floor in current training? | GNN approximation error (model quality), not Y-matrix |
| Is the `get_pypsa_Y_numpy` bug a risk? | **Yes** — if anyone uses `y_matrix_source='auto'`/`'pypsa'`, MSE floor = 270,000 for ieee30 |

### Experimental Proof (KCL residual with ground-truth voltages)

| Network | buses_i==buses_o | MSE_manual | MSE_buggy (PyPSA) | MSE_fixed |
|---|---|---|---|---|
| ieee9 (9B, PV=2, True) | True | 1.4e-15 ≈ **0** | 1.4e-15 | 1.4e-15 |
| ieee9 (9B, PV=2, False) | **False** | 5.8e-14 ≈ **0** | **5.6** | 5.8e-14 |
| cigre14 (18B, PV=2) | **False** | 1.7e-17 ≈ **0** | **0.9** | 1.7e-17 |
| ieee30 (30B, PQ+4PV) | **False** | 1.0e-21 ≈ **0** | **269,500** | 1.0e-21 |
| ieee30 (31B, PQ+4PV) | **False** | 9.4e-15 ≈ **0** | **56,370** | 9.4e-15 |

### Required Fix for `get_pypsa_Y_numpy` (Cell 8, both Training notebooks)

```python
# CURRENT (broken) — guard always False (str vs int), returns Y in buses_o order
sn_bus_order  = list(sn.buses_i())
net_bus_order = list(range(len(net_copy.buses)))
if sorted(sn_bus_order) == net_bus_order:
    return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]
else:
    return Y_dense   # ← buses_o order — WRONG for all non-trivial networks

# FIXED — reorder from buses_o → buses_i (network.buses.index order)
sn_buses_i = list(sn.buses_i())
sn_buses_o = list(sn.buses_o)
if sn_buses_i == sn_buses_o:
    return Y_dense
pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
perm = [pos_in_o[b] for b in sn_buses_i]
return Y_dense[np.ix_(perm, perm)]
```

### Why Current Training Is Protected

`run_hparam_sweep` (Cell 18) has `y_matrix_sources=("manual",)` as its default. All S1–S9 sweep cells use this default, so `compute_admittance_matrix` (buses_i order, correct) is always used. The bug is latent but not active.

### Missing Shunts — Not an Issue for Current Dataset

`compute_admittance_matrix` omits `network.shunt_impedances`. But ieee9, ieee30, and cigre14 all have **no shunt CSV files** — shunt loops are skipped. (ieee39 has shunts but is not in `mixed_1500_wide`.) This is a future concern only.

### Red Herrings Eliminated

| Hypothesis | Status |
|---|---|
| Stale `x_pu` in `get_pypsa_Y_numpy` | False — `calculate_Y()` calls `calculate_dependent_values()` internally |
| `v_ang` in degrees | False — training code uses radians; diagnostic scripts were wrong |
| Missing shunts causing floor | False — no shunts in active dataset |
| Unknown Q at PV/slack causes floor | False — `compute_power_flow_residual_from_pred` uses predicted Q/P, not zeroed inputs |

---

---

## Key Findings

### 1. Where the Physics Loss Is Computed

- **Location**: `GNN_Powerflow_V2.6_Training.ipynb` / `GNN_Powerflow_V2.6.1_Training.ipynb`, cell 11
- **Function**: `compute_power_flow_residual_from_pred(pred, x, Y_matrix, network, ...)`
- **What it does**: Assembles mixed state (known values from `x`, predicted values from `pred`), computes `S_calc = V * conj(Y * V)` using split real/imaginary Y-matrices, returns `mean(|P_calc - P_inj|² + |Q_calc - Q_inj|²)` per graph, averaged over the batch.
- **Why it matters**: The physics loss residual is `||S_calc(V_pred) - S_inj||²`, which goes to zero only if both (a) V predictions are perfect AND (b) the Y-matrix used matches the true network admittance.

---

### 2. Why the Y-Matrix Differs (Manual vs PyPSA)

**Location**: Cell 8 in Training notebook — `compute_admittance_matrix()` (manual) and `get_pypsa_Y_numpy()` (PyPSA extraction).

**Three independent sources of discrepancy:**

#### 2a. Missing ShuntImpedance components (confirmed)
- `compute_admittance_matrix` loops over `network.lines` and `network.transformers` only.
- It does **not** process `network.shunt_impedances`.
- PyPSA's `sn.calculate_Y()` **does** include shunt compensators in the Y-diagonal.
- **Impact per system**:
  | System | Shunts | Impact |
  |--------|--------|--------|
  | ieee9 | 0 | None — Y matrices match |
  | ieee30 | 0 (none exported) | Minimal |
  | cigre14 | 1 (capacitor bank) | Diagonal error at that bus |
  | ieee57 | 3 | Off by 3 diagonal entries |
  | ieee118 | ~14 | Significant diagonal errors |
- **For S1/S2 (ieee9 only)**: this is NOT the cause of the observed floor — ieee9 has no shunts.

#### 2b. Transformer shunt π-model terms (partial)
- The transformer loop explicitly says `# no shunt for now` — transformer shunt B_m (magnetizing susceptance) is omitted.
- For the flat-pu system with `tap_ratio=1.0`, transformer series admittance reduces to the same formula as a line, so this is only an issue if magnetizing admittance is non-negligible.
- PyPSA's `calculate_Y()` also typically omits magnetizing branches for static power flow transformers unless explicitly set.
- **Impact**: Likely negligible for the CSV-based systems in use.

#### 2c. Bus reordering fragility in `get_pypsa_Y_numpy` (potential bug)
- The function checks `if sorted(sn_bus_order) == net_bus_order` where `net_bus_order = list(range(len(net_copy.buses)))`.
- This check tests whether the subnetwork bus indices are the integer positions 0..N-1 in the same sorted order — it passes trivially for any connected network where PyPSA assigns sequential integer positions.
- **If it fails** (e.g., after topology changes with non-sequential indexing), the function returns `Y_dense` without any reordering — potentially assigning Y rows/cols to the wrong buses.
- For topology-variant networks (extra buses added with string names like `"extra_0"`), the subnetwork ordering may diverge from the network bus list ordering.
- **The `y_matrix_source='auto'` logic in V2.6.1** compares manual vs PyPSA and uses PyPSA when available. But if the PyPSA extraction silently returns a mis-ordered Y, the comparison will show a large diff (`n_diff += 1`) and **still use PyPSA** — with the wrong ordering.

---

### 3. Why a Wrong Y-Matrix Creates an Irreducible Physics Loss Floor

Let $\hat{Y}$ be the Y-matrix used in the physics loss, $Y_{true}$ be the correct PyPSA Y-matrix, $V_{true}$ be the true voltages from the AC power flow solution.

At the theoretical optimum (perfect predictions $\hat{V} = V_{true}$):

$$\text{physics\_loss} = \left\| \hat{Y} V_{true} \circ V_{true}^* - S_{true} \right\|^2$$

Since $Y_{true} V_{true} \circ V_{true}^* = S_{true}$ by definition (the AC PF solution satisfies KCL):

$$\text{physics\_loss} = \left\| (\hat{Y} - Y_{true}) V_{true} \circ V_{true}^* \right\|^2$$

This is **non-zero** whenever $\hat{Y} \neq Y_{true}$, regardless of how accurate the voltage predictions are. **The model cannot drive this term to zero.** The floor magnitude is proportional to the element-wise error in the Y-matrix and the typical voltage magnitudes (~1 pu).

For ieee9, if the Y-matrix is correct (`y_matrix_source='auto'` should pick up PyPSA's Y), this term should be near zero. The observed floor in S1/S2 therefore comes from the coupling mechanism in finding 4.

---

### 4. The Mathematical Physics Loss Floor (Jacobian Coupling)

Even with the **correct** Y-matrix, there is an irreducible physics floor for imperfect predictions.

For small prediction errors $\varepsilon = \hat{V} - V_{true}$, a first-order expansion gives:

$$\text{physics\_loss} \approx \| J \varepsilon \|^2$$

where $J$ is the AC power flow Jacobian. This means:

$$\text{physics\_floor} \approx \|J\|_F^2 \times \text{MSE\_floor}$$

**Key consequence**: The physics floor and MSE floor are mathematically coupled. As MSE converges to its floor, the physics loss converges to a proportional floor. Neither can be independently driven to zero.

The **adaptive S2 scheme** sets `eff_w_phys = fraction_physics × MSE / physics` to target a fixed ratio. But since `physics ≈ ||J||² × MSE` at steady state, the adaptive weight converges to `fraction_physics / ||J||²` — a constant. The model doesn't escape the floor; it just arrives at the same floor with a constant effective weight. This is why S1 and S2 show the same floor level.

---

### 5. Why Physics Loss Hurts More Than Helps at the Floor

From the earlier factorial analysis (`research_factorial_scaling_2026_05_19.md`), `weight_physics` has a positive coefficient — higher weight = worse accuracy. This has a clear mechanism:

**Before the floor**: The physics gradient provides signal complementary to MSE — it pushes predictions toward KCL-satisfying solutions. This is useful because KCL is a strong structural constraint.

**At the floor** (mixed-network training): The floor condition is `||J_i · ε_i||² ≈ const` for each topology $i$. The physics gradient is now $\nabla_\theta (\sum_i ||J_i \varepsilon_i||²)$. For a mixed set of topologies:
- Each topology has a different $J_i$ 
- The physics gradient direction becomes an average over conflicting Jacobians
- This average gradient does not point toward better MSE for any specific topology
- **Net effect**: physics gradient introduces noise relative to the MSE gradient, hurting convergence

The adaptive scheme does not fix this because the noise is structural (different $J_i$ matrices), not a scaling issue.

---

### 6. MSE Floor — Capacity or Training Issue?

**MSE floor for mixed training is expected and multi-causal:**

| Cause | Fixable? | Evidence |
|-------|----------|---------|
| Model capacity limit | Partially — `hidden_dim=128` helps | Factorial: `hidden_dim` largest negative coefficient |
| Mixed-topology generalization | No — fundamental | Same floor for all physics weights |
| Physics loss conflict | Yes — `w_phys=0` gives lower MSE | Factorial: positive `weight_physics` coefficient |
| Optimizer plateau | Partially — LR schedule | Not yet investigated |

**Is MSE ≈ 0 achievable?**

- **Single fixed topology, sufficient training**: YES, in principle. The AC PF mapping is deterministic and smooth; a GNN with enough capacity can overfit to near-0 training MSE.
- **Mixed training (1500 scenarios across topology variants)**: NO practical near-zero. The model must generalize across topologies using only graph structure. The information bottleneck (graph → GNN → output) means some irreducible generalization error.
- **Why it's NOT mainly a capacity issue**: From the research, `hidden_dim=128` vs `64` halves the floor, but doesn't approach 0. Adding more layers beyond 3 shows diminishing returns. The floor is a generalization floor, not a capacity ceiling for this architecture class.

**Practical implication**: An MSE floor of ~1e-3 to ~1e-4 (in pu) for mixed training is expected and acceptable. The corresponding physics floor is ~10-100× higher due to the Jacobian coupling.

---

### 7. Why PV Bus Q Residual Contributes to the Floor

In `compute_power_flow_residual_from_pred` with `use_q_partial_mode=False` (default):
- PV buses contribute Q residual: `|Q_calc(V_pred) - Q_pred|²`
- Q at PV buses is reactive power **absorbed** by the generator — it's a function of the network solution, not a "known" input
- The model predicts Q_pv independently; the physics loss compares this against Y-bus Q calculation
- These are consistent only when the model produces internally consistent (V_pv, Q_pv) pairs
- For the mixed training case, the model must learn this consistency for every topology variant

**`use_q_partial_mode=True`** would exclude PV Q from the residual, removing this contribution to the floor. This is the more physically correct interpretation: PV buses have free Q (within generator Q limits), so the physics constraint is only on P (known) and voltage magnitude (known). Enabling this may reduce the physics floor without changing the MSE floor.

---

## Key Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `GNN_Powerflow_V2.6.1_Training.ipynb`, cell 11 | `PhysicsConfig`, `compute_power_flow_residual_from_pred`, `compute_admittance_matrix` | All physics loss math |
| `GNN_Powerflow_V2.6.1_Training.ipynb`, cell 8 | `precompute_Y_matrices`, `get_pypsa_Y_numpy` | Y-matrix source selection |
| `GNN_Powerflow_V2.6.1_Training.ipynb`, cell 14 | Training loop — adaptive/fixed weight dispatch | S1 vs S2 behavior |
| `GNN_Powerflow_V2.6.1_Training.ipynb`, cells 24/27 | S1 and S2 sweep config definitions | Exact configs run |
| `.docs/research/research_factorial_scaling_2026_05_19.md` | Factorial analysis — physics coefficient sign | Supporting evidence for "physics hurts" |

---

## Constraints & Considerations

- The `y_matrix_source='auto'` in V2.6.1 should use PyPSA's Y for most ieee9 topologies — but the bus reordering code may silently mis-order the Y for topology-variant networks with extra buses.
- `use_q_partial_mode=False` is a conservative choice that includes more physics constraints but also more floor noise from PV bus Q.
- For single-topology training, the physics loss CAN be driven very close to zero — it is only problematic in the mixed setting.
- The S2 adaptive scheme effectively becomes a constant-weight scheme at steady state; it does not provide qualitatively different behavior from S1.

---

## Recommendations

1. **Verify Y-matrix correctness for topology-variant networks**: Run a diagnostic that prints `n_diff` (mis-matched networks) from `precompute_Y_matrices` for the mixed_1500 training set. If high, the bus reordering bug is active.

2. **Try `use_q_partial_mode=True`**: Should reduce the PV-bus contribution to the physics floor without hurting MSE. Low-effort, high-signal experiment.

3. **Physics loss as a warm-start, not steady-state regularizer**: Apply physics weight only during warmup (epochs 0-10 with ramp), then set `w_phys=0` for the remainder. This provides early structural guidance without the steady-state conflict. Warmup-only physics can be implemented as `cfg_fixed_04` with `warmup_epochs=30, num_epochs=30` (all warmup, zero after).

4. **Log `n_diff` and `n_manual` from `precompute_Y_matrices`**: Add these counts to `run_info` so sweeps can diagnose Y-matrix quality per run.

5. **For near-zero MSE**: Train on a single fixed topology (no variants) with high capacity (`hidden_dim=128, num_layers=4`). This should achieve MSE ~1e-5 or better and confirm the mixed-training floor is a generalization phenomenon, not a model capacity issue.

---

## Open Questions

- [ ] For the actual S1/S2 runs: what are the numerical floor values (physics floor ≈ ? × MSE floor)? This ratio should equal `||J_avg||²`.
- [ ] Is the bus reordering bug in `get_pypsa_Y_numpy` actually triggered for the mixed_1500 topology-variant networks?
- [ ] Does `use_q_partial_mode=True` measurably reduce the physics floor in an ablation?
- [ ] At what epoch does the physics loss plateau vs MSE? If physics floors first (before MSE), that confirms Y-matrix mismatch dominates. If they floor together, Jacobian coupling is the dominant mechanism.
