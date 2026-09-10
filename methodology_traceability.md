# Methodology Traceability — DC Baseline Reactive-Power Reporting

**Generated:** 2026-09-07
**Scope:** Single methodology slice (baseline evaluation / reporting conventions), not the full pipeline.
**Source notebook:** `GNN_Powerflow_V2.7_Analysis.ipynb`
**Paired manuscript file:** `methodology_generated.tex` (subsection "Reactive-power reporting for the DC baseline")

---

## 1. Scope

Documents the reporting-consistency fix applied to the DC (linearised) power-flow baseline's
reactive-power metrics, and the corresponding visual-disambiguation convention in the
timing-versus-accuracy comparison figures. Does not cover the rest of the evaluation pipeline
(GNN architecture, training loss, other baseline solvers) which are unchanged.

---

## 2. Evidence Mapping

| Step ID | Method Step (paper wording) | Implementation Evidence (file/symbol/notebook section) | Dependencies/Config | Assumptions/Limitations |
|---------|------------------------------|----------------------------------------------------------|----------------------|--------------------------|
| M1 | DC baseline solves angle + active-power balance only; voltage magnitude flat at 1 p.u., no reactive-power model. | `evaluate_dc_baseline`, cell "Comparison & Solver Baselines" — calls `network.lpf()` (PyPSA linear power flow); `vmag_dc = np.ones(len(buses))`. | PyPSA `Network.lpf()` | Flat-voltage / lossless-DC assumption is inherent to PyPSA's linear power-flow solver, not a project-specific approximation. |
| M2 | Bus-level reactive-power injection error is scored against an assumed value of zero (pre-existing behaviour, unchanged by this revision). | `evaluate_dc_baseline`: `q_dc = np.zeros(len(buses))`, compared to `q_true` via `all_q_err`. | — | Zero-guess only applies to buses where Q is unknown to DC (PV/Slack); PQ buses use the known injected Q, not a guess. |
| M3 | Branch-level reactive-flow error is now scored under the same zero-guess convention (previously reported as not-applicable). | `evaluate_dc_baseline`: new `all_lflow_q0_err` accumulator; `true_q0 = network.lines_t.q0.loc[snap].values; all_lflow_q0_err.extend(np.abs(true_q0))` (guess = 0). Metrics dict keys `line_flow_q0_mae` / `line_flow_q0_rmse` now computed from this accumulator instead of hardcoded `float("nan")`. | Requires `network.lines_t.q0` (PyPSA post-solve reactive line flow) to be populated. | If `lines_t.q0` is empty for a given snapshot, that snapshot is silently skipped from the aggregate (no fallback value). |
| M4 | DC-baseline points on reactive-power metric panels are visually flagged as a naive guess, distinct from genuine power-flow solver points (PyPower, pandapower, FDLF, full AC PF). | `plot_timing_vs_accuracy`: `_DC_Q_GUESS_METRICS = {"q_mae", "q_rmse", "line_flow_q0_mae", "line_flow_q0_rmse"}`; marker set to `"X"` (vs `"*"` for real solves) and label suffixed `" (Q=0 guess)"` when `dc_label` starts with "dc" (case-insensitive) and the current panel metric is in `_DC_Q_GUESS_METRICS`. Legend gets an added entry `"DC Q=0 guess (naive)"`. | Baseline dict keys passed via `ref_baselines`/`ref_baselines_batch` (built by `build_ref_baselines`) must be labelled with a name starting "DC" (e.g. `"DC lpf"`) for the flag to trigger. | Detection is a string-prefix heuristic on the baseline's display label, not a structural flag on the metrics dict; renaming the DC baseline entry to something not starting with "DC" would silently disable the visual flag. |

---

## 3. Related (non-methodological) change — not included in manuscript text

| Item | Description | Rationale for exclusion from `.tex` |
|------|-------------|---------------------------------------|
| Leader lines on plot annotations | `plot_timing_vs_accuracy` now passes `arrowprops=dict(arrowstyle="-", ...)` to `ax.annotate(...)` for both per-run and baseline labels, drawing a thin connector line from each scatter point to its text label. | Purely a figure-readability improvement; does not affect any reported quantity, metric definition, or scientific claim. Mention only in a figure caption if desired, not in methodology prose. |

---

## 4. Open Questions / Ambiguities

- None outstanding for this slice. Both accumulator naming (`all_lflow_q0_err`) and the metrics-dict keys it feeds (`line_flow_q0_mae`, `line_flow_q0_rmse`) were verified in the same edit pass, so no partial/mismatched state exists in the current notebook.
- The "DC" label-prefix heuristic (M4) is a latent fragility worth flagging to the user if the baseline-naming convention is ever changed; not fixed here since it is a plotting-robustness concern, not a methodology-correctness concern.
