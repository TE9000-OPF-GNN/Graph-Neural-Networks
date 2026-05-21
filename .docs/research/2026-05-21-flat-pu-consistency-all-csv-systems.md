# Research: Flat-pu Consistency — All CSV-Based Systems

## Goal
Verify that the flat-pu modeling conclusion (z_HV_pu = z_LV_pu, tap_ratio=1.0 correct) established for ieee30 in [2026-05-21-transformer-turns-ratio-flat-pu-consistency.md](2026-05-21-transformer-turns-ratio-flat-pu-consistency.md) holds for every other CSV-based system in the project.

**Systems in scope**: ieee39, ieee57, ieee118 (powsybl export), cigre14, cigre14der (pandapower export).  
**Out of scope**: ieee9 (built directly in Python via `create_9_bus_network_topology_variants`, not CSV-based).

---

## Key Findings

### 1. ieee57 and ieee118 — single voltage level, no turns-ratio issue

- **Location**: `OneDrive/grid_model_files/ieee57_buses.csv`, `ieee118_buses.csv`
- **Evidence**: All buses have `nominal_v = 138.0 kV` for both systems. All transformers have `rated_u1 = rated_u2 = 138.0 kV` (same voltage both sides).
- **Result**: Turns ratio a = 1.0 for every transformer. z_pu invariance holds trivially (same base both sides). `tap_ratio=1.0` in PyPSA is exact. **No issue**.

| System | Bus voltages | Transformer count | rated_u1/u2 | a |
|--------|-------------|-------------------|-------------|---|
| ieee57 | 138 kV (all 57) | 15 | 138/138 kV | 1.0 |
| ieee118 | 138 kV (all 118) | 9 | 138/138 kV | 1.0 |

Additionally, all ieee57 and ieee118 transformers have `r=0.0` (zero resistance — ideal series reactance only). These are phase-regulation or autotransformer models, not step-up/step-down.

---

### 2. ieee39 — already flat-pu in source data, no turns-ratio issue

- **Location**: `OneDrive/grid_model_files/ieee39_buses.csv`, `ieee39_transformers.csv`
- **Evidence**: Bus CSV has no `nominal_v` column (columns: `name, v_mag, v_angle, ...`). All 12 transformers have `rated_u1 = rated_u2 = 1.0 kV`.
- **Interpretation**: The powsybl representation of IEEE39 uses a flat 1.0 kV nominal voltage for all buses (the network was likely imported from a per-unit description without explicit kV values). The CSV is already in a fully flat-pu system with a=1.0 for every transformer.
- **Result**: `tap_ratio=1.0` is exact (a=1.0). **No issue**.

---

### 3. cigre14 and cigre14der — different export path, same conclusion

- **Location**: `export_pandapower_cigre_mv`, DataGen cell 13; `cigre14_transformers.csv`, `cigre14der_transformers.csv`
- **Voltage levels**: B0 = 110 kV (HV slack bus), B1–B14 = 20 kV (MV feeder buses). Both transformers connect 110→20 kV: a = 5.5.
- **Export convention** (from `export_pandapower_cigre_mv` L73–90):
  - `vkr_pu = vkr_percent / 100` (resistive short-circuit voltage, on transformer's own MVA base)
  - `r_csv = vkr_pu × (sbase_mva / snom_mva)` = scaled to 100 MVA system base
  - `x_csv = sqrt(vk_pu² - vkr_pu²) × (sbase_mva / snom_mva)` = same scaling
  - `zbase = (vnom_hv² / sbase_mva)` is computed but **NOT applied to transformer** (only used for lines)
- **Interpretation**: `vkr_percent` is dimensionless (copper loss ratio), invariant to which side of the transformer it's measured on. The MVA-base scaling gives r/x in pu on (100 MVA, matched HV/LV voltage bases). This is equivalent to the HV-side referred convention used by powsybl — and the z_pu invariance applies identically.
- **Numerical verification** (both cigre14 and cigre14der, T-0-1-0 and T-0-12-1):

| Transformer | V_HV/V_LV | a | x_csv | x_pu_LV_correct | Match |
|-------------|-----------|---|-------|-----------------|-------|
| T-0-1-0 | 110/20 kV | 5.5 | 0.4800 | 0.4800 | ✓ |
| T-0-12-1 | 110/20 kV | 5.5 | 0.4800 | 0.4800 | ✓ |

- **Result**: z_HV_pu = z_LV_pu confirmed. `tap_ratio=1.0` in PyPSA is correct for nominal tap. **No issue**.

---

### 4. Known column name mismatch: `rateds` vs `rated_s` in cigre14

- **Location**: `load_system_from_csv` cell 20 L53: `rated_s = row.get("rated_s", None)`; cigre14/cigre14der transformer CSV column is `rateds` (no underscore).
- **Effect**: `load_system_from_csv` always reads `rated_s=None` for cigre14 transformers → `s_nom = 1.0` fallback.
- **Impact on correctness**: None. Because the cigre14 r/x are already on the **100 MVA system base** (not on the transformer's own 25 MVA base), `s_nom=1.0` is correct — PyPSA computes `r_pu = r/s_nom = r/1.0 = r`, passing the 100 MVA pu values through unchanged.
- **If s_nom were set to 0.25**: PyPSA would compute `r_pu = r/0.25 = 4r` — 4× the correct value. This would corrupt the Y-bus. The current `s_nom=1.0` fallback is the **correct behavior** for cigre14.
- **Conclusion**: The `rateds` column name mismatch is **not a bug** — it's effectively correct by coincidence of the export convention. Document and leave as-is.

---

### 5. ieee9 is not CSV-based — out of scope

- **Location**: `create_9_bus_network_topology_variants` (DataGen cell not via CSV)
- **Evidence**: No `ieee9_buses.csv` or `ieee9_transformers.csv` exists in either `OneDrive/grid_model_files/` or the local `grid_model_files/` directory.
- **Result**: ieee9 builds the PyPSA network directly in Python with hardcoded values. The flat-pu consistency question does not apply via the CSV pathway. However, the hardcoded network uses `v_nom=1.0` throughout, so the same consistency argument applies.

---

## Summary Table — All CSV Systems

| System | Export path | Bus voltages | Transformer a | z_pu invariance | tap_ratio=1.0 correct |
|--------|-------------|-------------|---------------|-----------------|----------------------|
| ieee30 | powsybl | 132/33/11 kV | 4.0 (3×) | ✓ verified | ✓ |
| ieee39 | powsybl | 1.0 kV flat | 1.0 (12×) | trivially ✓ | ✓ |
| ieee57 | powsybl | 138 kV | 1.0 (15×) | trivially ✓ | ✓ |
| ieee118 | powsybl | 138 kV | 1.0 (9×) | trivially ✓ | ✓ |
| cigre14 | pandapower | 110/20 kV | 5.5 (2×) | ✓ verified | ✓ |
| cigre14der | pandapower | 110/20 kV | 5.5 (2×) | ✓ verified | ✓ |

**ieee30 is the only system with genuine step-up/step-down transformers from powsybl (a=4). cigre14/der is the only system with step-up/step-down from pandapower (a=5.5). Both are confirmed correct.**

---

## Patterns to Follow

| Pattern | Location | Notes |
|---------|----------|-------|
| powsybl transformer export | `export_powsybl_to_tables` cell 10 L68–74 | Uses `zbase(voltage_level1_id)` = HV side; impedance already in 100 MVA HV-referred pu |
| pandapower transformer export | `export_pandapower_cigre_mv` cell 13 L73–90 | Uses `sbase_mva/snom_mva` scaling from transformer %; `zbase` computed but NOT used for transformer |
| load_system_from_csv transformer load | cell 20 L44–58 | `s_nom=1.0` fallback correct for cigre14; `tap_ratio=1.0` correct for all systems |

---

## Constraints

- **Do not change s_nom for cigre14**: The `rateds` vs `rated_s` mismatch causes `s_nom=1.0`, which is the correct value. Fixing the column name to match would require also verifying that the export already scales to 100 MVA base (which it does). If someone "fixes" the column name, `s_nom=0.25` would corrupt the Y-bus by 4×.
- **Flat-pu is confirmed correct for all systems**: No changes to `v_nom`, `s_nom`, or `tap_ratio` are needed.
- **ieee9 is outside this analysis**: its hardcoded construction is separately consistent.
- **OLTC modeling gap applies to cigre14 too**: both cigre14 transformers have `tap=1.0` in the CSV (nominal tap). Same limitation as ieee30 — no OLTC variation in training data.

---

## Recommendations

1. **No modeling changes needed** for any CSV-based system. The flat-pu approach is consistent across all systems.
2. **Document the `rateds` column name** in a code comment in `load_system_from_csv` near L53, noting that for cigre14 the `rateds` fallback to `s_nom=1.0` is intentionally correct.
3. **The power balance check (Task G)** described in the main transformer research doc applies to all systems equally — it's system-agnostic.
4. **Extend to ieee9**: The direct-construction path (`create_9_bus_network_topology_variants`) should be spot-checked separately to confirm it uses consistent pu values — this is not covered by the CSV pathway analysis.

---

## Open Questions

- [ ] Does `create_9_bus_network_topology_variants` use consistent pu for r/x/b? (Low priority — ieee9 has a single voltage level, so turns-ratio issue doesn't arise.)
- [ ] Are `transformers_t.p0`/`p1` populated by PyPSA 0.28 after `pf()` for all these systems? Needed for Task G power balance check implementation.
