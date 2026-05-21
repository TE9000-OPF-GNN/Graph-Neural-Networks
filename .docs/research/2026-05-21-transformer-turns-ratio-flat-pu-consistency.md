# Research: Transformer Turns Ratio & Flat-pu Consistency Check

## Goal
Determine whether the current flat-pu system (`v_nom=1.0` everywhere, `tap_ratio=1.0` for all transformers, impedances from Powsybl using actual kV bases) gives a physically correct power flow solution for multi-voltage networks (ieee30: 132/33/11 kV levels), and specify a visible consistency check to add to the data generation pipeline.

**Scope**: `GNN_Powerflow_V2.6_DataGen.ipynb`, `export_powsybl_to_tables` (cell 10), `load_system_from_csv` (cell 20), PyPSA 0.28 pf.py internals.

---

## Key Findings

### 1. Transformer pu impedance is the same on both sides — z_HV_pu = z_LV_pu

- **Theory**: In a consistent per-unit system where voltage bases are matched to the nominal turns ratio (`V_base_LV = V_base_HV / a`), the pu impedance is invariant:
  - `z_base_HV = V_HV² / S_base`, `z_base_LV = V_LV² / S_base = V_HV² / (a² S_base)`
  - `z_base_HV = a² × z_base_LV`
  - `z_pu_HV = z_Ohm / z_base_HV`, `z_pu_LV = (z_Ohm / a²) / z_base_LV = z_Ohm / (a² × z_base_LV) = z_Ohm / z_base_HV = z_pu_HV` ✓

- **Evidence** (all 4 ieee30 transformers, `rated_u1`/`rated_u2` from CSV):

| Transformer | kV ratio | a | x_csv (HV pu) | x_pu_LV (correct) | Match |
|-------------|----------|---|---------------|-------------------|-------|
| T-4-12-1 | 132/33 | 4.0 | 0.01600 | 0.01600 | ✓ |
| T-6-10-1 | 132/33 | 4.0 | 0.03475 | 0.03475 | ✓ |
| T-28-27-1 | 132/33 | 4.0 | 0.02475 | 0.02475 | ✓ |
| T-6-9-1 | 132/1 | 132 | 0.000012 | 0.000012 | ✓ |

- **Location**: `export_powsybl_to_tables` cell 10, L58: `zb = zbase(row.get("voltage_level1_id", ""))` → uses HV voltage level. The key `vnom_kv` is built from `net.get_voltage_levels()` (L24), correctly mapping "VL4" → 132 kV etc.

---

### 2. PyPSA Y-bus is correct for nominal-tap transformers with flat v_nom

- **Location**: `pypsa/pf.py` L1242–1264 (`_calculate_Y_bus`)

```python
y_se = 1 / (branches["r_pu"] + 1j * branches["x_pu"])
tau   = branches["tap_ratio"].fillna(1.0)     # all 1.0 in our code
tau_hv = 1.0 for tap_side≠0, else tau
tau_lv = 1.0 for tap_side≠1, else tau

Y00 = (y_se + 0.5*y_sh) / tau_hv**2    # HV bus (bus0)
Y11 = (y_se + 0.5*y_sh) / tau_lv**2    # LV bus (bus1)
Y10 = -y_se / (tau_hv * tau_lv)         # off-diagonal
```

- **With `tau=1.0`**: Y00 = Y11 = y_se, Y10 = -y_se. This is the symmetric π-model.
- **Why this is correct**: Since z_pu is the same on both sides (Finding 1), y_se = 1/z_pu gives the correct admittance contribution to BOTH the HV bus and the LV bus. The symmetric Y-bus is the physically correct model for a nominal-tap transformer in a per-unit system with matched bases.
- **Standard reference**: This is identical to MATPOWER's `makeYbus` (pf.py comment L1237). MATPOWER uses the same convention: off-nominal tap transformers use τ≠1, nominal-tap transformers use τ=1 and appear as simple series impedances.

---

### 3. Power flow solution is numerically correct; only kV interpretation differs

- **In proper multi-voltage pu**: 1.02 pu at a 132 kV bus = 134.6 kV; 0.98 pu at a 33 kV bus = 32.3 kV. Physical voltage ratio across transformer = 4.17 (close to nominal 4:1).
- **In flat pu (v_nom=1.0)**: 1.02 pu at any bus = 1.02 × 1 kV. The "physical" voltage ratio = 1.04 — clearly wrong as a kV value.
- **But**: The power flow equations `I = Y × V`, `P+jQ = V·conj(I)` are identical in both systems — the same y_se, same P/Q injections, same solution V_pu. The numerical values of V_pu, θ, P_pu, Q_pu are identical.
- **For GNN training**: All node features (v_mag_pu, v_ang, P_pu, Q_pu) and edge features (r_pu, x_pu, b_pu) are in the flat pu system. The GNN learns in this system and makes predictions in this system. Physical kV values are never used. **The training data is consistent and correct.**

---

### 4. One genuine gap: off-nominal tap (OLTC) positions not modeled

- **All current transformers**: `tap_ratio=1.0` hardcoded in `load_system_from_csv` (cell 20, L57).
- **Real transformers**: OLTC controllers may operate at ±5–10% of nominal tap. For a 5% off-nominal tap: τ=1.05, Y00 = y_se / 1.05² = 0.907 y_se (9.3% change vs 1.0).
- **Powsybl CSV**: `rho` column stores the tap ratio, `r_at_current_tap`/`x_at_current_tap` store impedances at the operating tap position. **These are not currently read.**
- **Impact**: Training data uses only nominal-tap operating points for all transformers. OLTC variation is absent from the training distribution. Whether this matters depends on the use case (nominal-tap only = acceptable; OLTC scenarios = gap).

---

### 5. Proposed consistency check: system power balance after PF

- **What to check**: After each successful power flow, compute the system-level power balance residual:
  ```
  ΔP = |total_gen_P − total_load_P − total_line_losses| / max(|total_gen_P|, ε)
  ```
  Where `total_line_losses = |lines_t.p0 + lines_t.p1|.sum() + |trafos_t.p0 + trafos_t.p1|.sum()`.
- **Alternatively (simpler via PyPSA)**: use `network.buses_t.p` (net injection per bus per snapshot) if available, or compute from generator/load contributions.
- **Threshold**: ΔP < 1e-3 (0.1% of total generation). PyPSA's NR typically achieves 1e-6 or better.
- **Where to add**: Extend `sanity_check_power_flow` (cell 17) with a new check block:
  - Input: `network` (post-pf), `max_balance_err: float = 1e-3`
  - Check 2c (add after existing voltage/angle checks):
    ```python
    # Check 2c: system power balance
    gen_p = network.generators_t.p.values.sum()
    load_p = network.loads_t.p.values.sum()
    line_loss = (network.lines_t.p0.values + network.lines_t.p1.values).sum()
    traf_loss = (network.transformers_t.p0.values + network.transformers_t.p1.values).sum()
    balance_err = abs(gen_p - load_p - line_loss - traf_loss) / max(abs(gen_p), 1e-6)
    if balance_err > max_balance_err:
        raise RuntimeError(f"Power balance error {balance_err:.2e} exceeds {max_balance_err}")
    ```
- **Expected values**: For well-converged AC PF, balance_err < 1e-5. Errors > 1e-3 indicate a modeling inconsistency or failed convergence.

---

## Patterns to Follow

| Pattern | Location | Notes |
|---------|----------|-------|
| Existing sanity checks | `sanity_check_power_flow`, cell 17 | Extend with balance check, same try/except structure |
| Post-PF result access | `network.generators_t.p`, `network.lines_t.p0/p1` | Available after `network.pf()` |
| Check function signature | `sanity_check_power_flow(network, base_system, ...)` | Add `max_balance_err=1e-3` param |

## Key Files

| File / Location | Purpose | Relevance |
|-----------------|---------|-----------|
| `export_powsybl_to_tables`, cell 10 L22–64 | Computes z_pu using actual kV bases | Establishes pu convention |
| `load_system_from_csv`, cell 20 L41–58 | Loads transformers with `tap_ratio=1.0` | Confirms flat-pu setup |
| `pypsa/pf.py` L1242–1264 | Y-bus assembly with `tau` | Confirms tap_ratio usage |
| `sanity_check_power_flow`, cell 17 L155–215 | Existing post-PF validation | Where to add balance check |

---

## Constraints

- **Consistency of flat-pu is confirmed**: Do not change `v_nom`, `s_nom`, or `tap_ratio` independently. The system is self-consistent as-is (per [2026-05-21-sn-mva-s-nom-v-nom-safety-check.md](2026-05-21-sn-mva-s-nom-v-nom-safety-check.md)).
- **OLTC modeling is out of scope for now**: Adding tap variation would require reading `rho` from the transformer CSV and a new topology-variation step. This is a future enhancement.
- **T-6-9-1 (132→1 kV artifact)**: This transformer should not affect conclusions — it is likely a powsybl model artifact. Its z_pu invariance holds numerically but the network itself is physically questionable (see negative-rxb research doc Finding 8).
- **Transformer balance check needs `transformers_t`**: Only populated if network has transformers AND power flow converged. Guard with `hasattr(network, 'transformers_t')`.

---

## Recommendations

1. **No changes needed to core modeling**: The flat-pu system is correct for nominal-tap power flow. Do NOT change v_nom, s_nom, or tap_ratio.

2. **Add system power balance check to `sanity_check_power_flow`** (cell 17): New `max_balance_err=1e-3` parameter + check 2c block as specified in Finding 5 above. This makes the consistency visible at dataset generation time.

3. **Optionally add per-bus KCL check** (stronger but more complex): Compute injection residual at each bus using `buses_t` and branch flows. Useful for diagnosing which bus/branch has an inconsistency.

4. **Document OLTC gap**: Note in dataset metadata that all transformers are at nominal tap. If OLTC variation is needed in future, read `rho` from CSV and pass to `tap_ratio` in `load_system_from_csv`.

---

## Open Questions

- [ ] Does `sanity_check_power_flow` receive the post-PF network (after `pf()` is called), or is it called on the pre-PF network? Need to verify call site in `generate_training_data_with_topology`.
- [ ] Are `transformers_t.p0`/`p1` actually populated by PyPSA 0.28 after `pf()`? Check with a small test network.
- [ ] Is the VL9 artifact (T-6-9-1, x=0.000012 pu = near short-circuit) causing PyPSA's NR to struggle with ieee30? The near-zero impedance may cause the Y-bus to be ill-conditioned.
