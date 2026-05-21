# Plan 011 — BUGFIX: DataGen Physical Limits & pu Correctness

**Status**: Revised — research complete; tasks reordered by priority  
**Priority**: High — affects data quality of all newly generated training sets  
**Research artifacts**:
- [2026-05-21-negative-rxb-and-transformer-handling.md](../research/2026-05-21-negative-rxb-and-transformer-handling.md)
- [2026-05-21-sn-mva-s-nom-v-nom-safety-check.md](../research/2026-05-21-sn-mva-s-nom-v-nom-safety-check.md)
- [2026-05-21-transformer-turns-ratio-flat-pu-consistency.md](../research/2026-05-21-transformer-turns-ratio-flat-pu-consistency.md)

**Notebook**: `GNN_Powerflow_V2.6_DataGen.ipynb`

---

## Intention

Fix data quality issues in `load_system_from_csv` and `sanity_check_power_flow` that cause generated PyPSA networks to have non-physical parameters or lack visible post-solve validation. The core flat-pu modeling (`v_nom=1.0`, `tap_ratio=1.0`) is **confirmed correct** — do not change it.

---

## Tasks

### REVERT REQUIRED

- [ ] **F. REVERT Generator Q limits patch** — The `q_min_pu`/`q_max_pu` attributes added in commit `4dfe772` are **not valid PyPSA 0.28 Generator attributes**. PyPSA silently ignores them with a per-generator WARNING on every `add()` call. This generates ~100+ log lines per network and has zero effect on the power flow solution. **Action**: revert the patch to `load_system_from_csv` in cell 20 — remove the `q_min_raw`/`q_max_raw` block and `gen_kw["q_min_pu"]`/`gen_kw["q_max_pu"]` conditionals. Revert via surgical disk patch. Note: real Q limit enforcement in PyPSA would require a post-pf() PV→PQ bus-switching loop and is out of scope.

### PENDING — safe to implement

- [ ] **A. Clamp b to ≥ 0 in line loading** — `load_system_from_csv` does not clamp the `b` value read from CSV. Three lines in the stale local ieee30 CSV have negative b. Add defense-in-depth floor:
  ```python
  b_val = max(float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0), 0.0)
  ```
  Low risk. OneDrive CSVs already have b≥0 after the cross-VL fix; this is a guard for local stale files.

- [ ] **B. Set `n.sn_mva = 100.0`** — PyPSA default is `sn_mva=1.0`; all P/Q/Z in CSVs are on 100 MVA base. **Cosmetic only** — PyPSA does not use `sn_mva` in any power flow computation (confirmed in `pf.py` source: zero hits). But it makes diagnostic functions and MW/MVAr conversions show correct magnitude.
  ```python
  n = pypsa.Network()
  n.sn_mva = 100.0
  ```
  Safe to implement independently (does NOT affect impedance values or PF solution).

- [ ] **G. System power balance check in `sanity_check_power_flow`** — Add a visible KCL consistency check after successful PF. New `max_balance_err=1e-3` parameter:
  ```python
  # Check 2c: system power balance (KCL)
  gen_p  = network.generators_t.p.values.sum()
  load_p = network.loads_t.p.values.sum()
  line_loss = (network.lines_t.p0.values + network.lines_t.p1.values).sum()
  traf_p = (network.transformers_t.p0.values + network.transformers_t.p1.values)
  traf_loss = traf_p.sum() if traf_p.size > 0 else 0.0
  balance_err = abs(gen_p - load_p - line_loss - traf_loss) / max(abs(gen_p), 1e-6)
  if balance_err > max_balance_err:
      raise RuntimeError(f"Power balance error {balance_err:.2e}")
  ```
  Expected for well-converged AC PF: balance_err < 1e-5. Errors > 1e-3 indicate modeling inconsistency. Add `max_balance_err=1e-3` to function signature.

### DEFERRED / DO NOT IMPLEMENT AS DESCRIBED

- [~~C~~] ~~**Transformer s_nom default to 100.0**~~ — **DANGEROUS — do not implement.** Research confirmed: with the flat-pu system (`v_nom=1.0`, `s_nom=1.0`), PyPSA computes `r_pu = r / s_nom = r / 1.0 = r`. If `s_nom` is changed to 100 without rescaling `r`/`x`/`b`, the admittances become 100× larger (more conductive) — fundamentally corrupting the Y-bus and all PF solutions. The current `s_nom=1.0` is **intentional** and correct for the flat-pu convention. See [sn-mva-s-nom-v-nom-safety-check.md](../research/2026-05-21-sn-mva-s-nom-v-nom-safety-check.md).

- [~~D~~] ~~**Transformer v_nom from actual kV / tap_ratio from turns ratio**~~ — **DEFERRED.** Research confirmed the flat-pu system is correct: for nominal-tap transformers, `z_HV_pu = z_LV_pu` (verified numerically for all 4 ieee30 transformers), so `Y00 = Y11 = y_se` with `tap_ratio=1.0` is the mathematically correct Y-bus. The physical kV interpretation is wrong, but the numerical PF solution is identical to the proper multi-voltage system. Changing to actual kV would require re-normalising ALL line and transformer impedances — a full re-parametrisation that goes beyond a bugfix. OLTC variation (off-nominal tap) is a separate future enhancement. See [transformer-turns-ratio-flat-pu-consistency.md](../research/2026-05-21-transformer-turns-ratio-flat-pu-consistency.md).

---

## Constraints

- **Surgical disk patching protocol**: all changes to `.ipynb` must use Python patch scripts with `open(..., newline='\n')`. Always `git commit` checkpoint before patching. Verify with `git diff` before keeping.
- **No kernel-state changes**: patching `.ipynb` on disk does not update the in-kernel function. The kernel must re-execute cell 20 (`load_system_from_csv`) after any patch.
- **Flat-pu is correct — do not change it**: `v_nom=1.0`, `s_nom=1.0`, `tap_ratio=1.0` are all intentional and self-consistent. Research confirmed tasks C and D would corrupt the Y-bus.
- **Tasks B and G are independent**: B (`sn_mva`) has zero effect on PF; G (balance check) reads post-PF results. Either can be done alone.
- **Existing datasets**: not retroactively affected. Must regenerate for any structural changes to take effect.
- **CIGRE14 format**: uses pandapower export format — any new columns will be absent. All fallback paths handle this silently.

---

## Verification Steps

| Task | Verification |
|------|-------------|
| F (revert Q limits) | `grep "q_min_pu" GNN_Powerflow_V2.6_DataGen.ipynb` → zero hits; run cell 20, verify no `q_min_pu` warnings in PyPSA output |
| A (b clamp) | `python -c "import json; src=''.join(json.load(open('GNN_Powerflow_V2.6_DataGen.ipynb'))['cells'][19]['source']); print('b_val = max' in src)"` → True |
| B (sn_mva) | After re-running cell 20: `net.sn_mva == 100.0`; verify `net.transformers["s_nom"].values` unchanged (still ~1.0) |
| G (balance check) | After PF on ieee9: `sanity_check_power_flow(n, ...)` completes with no RuntimeError; balance_err logged < 1e-4 |
