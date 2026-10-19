"""Task 090: exec-based tests of the Contingency notebook's metrics/report and driver cells.

ALL Contingency code cells (incl. toggles and driver) are exec'd in order in a fresh namespace (IPython magics
commented out). Every metric is tested on synthetic, hand-built record frames (same schema as task 089/091), so no
dataset is needed; files are only written under pytest's tmp_path.
"""
# [STSI 141026]:new metrics/report tests for the contingency notebook (task 090)
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
CONTINGENCY = os.path.join(REPO, "GNN_Powerflow_V2.7_Contingency.ipynb")
LIB_BANNER = "# ==== CONTINGENCY: METRICS & REPORT (task 090) ===="
DRIVER_BANNER = "# ==== CONTINGENCY: DRIVER (task 090) ===="
LEVELS = ("violation", "severe", "contingency_induced")
LABELS = {
    ("bus", "under_v"): {"violation": "under_v", "severe": "severe_under", "contingency_induced": "induced_under"},
    ("bus", "over_v"): {"violation": "over_v", "severe": "severe_over", "contingency_induced": "induced_over"},
    ("line", "overload"): {"violation": "overload", "severe": "severe_overload",
                           "contingency_induced": "induced_overload"},
}


def _clean(src):
    out = []
    for line in src.split("\n"):
        st = line.lstrip()
        out.append(line[: len(line) - len(st)] + "#" + st if st.startswith(("%", "!")) else line)
    return "\n".join(out)


def _code_sources(path):
    with open(path, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in cells if c["cell_type"] == "code"]


def _exec_all(name):
    namespace = {"__name__": name}
    for i, src in enumerate(_code_sources(CONTINGENCY)):
        exec(compile(_clean(src), f"<contingency:code{i}>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture(scope="module")
def ns():
    return _exec_all("__task090_test__")


# ── synthetic record builders (task 089 schema, default bands 0.90/1.10 and 0.85/1.15, severe factor 1.2) ──
def _bus_frame(entries, pre=None, status=None):
    """entries: (case_id, system, snapshot, bus, vmag); pre: per-row (pre_under, pre_over)."""
    df = pd.DataFrame(entries, columns=["case_id", "system", "snapshot", "bus", "vmag"])
    v = df["vmag"]
    df["under_v"], df["over_v"] = v < 0.90, v > 1.10
    df["violation"] = df["under_v"] | df["over_v"]
    df["severe_under"], df["severe_over"] = v < 0.85, v > 1.15
    df["severe"] = df["severe_under"] | df["severe_over"]
    pu, po = (np.zeros(len(df), bool),) * 2 if pre is None else np.asarray(pre, bool).T
    df["induced_under"], df["induced_over"] = df["under_v"] & ~pu, df["over_v"] & ~po
    df["contingency_induced"] = df["induced_under"] | df["induced_over"]
    if status is not None:
        df["status"] = status
    return df


def _line_frame(entries, pre_overload=None, status=None):
    """entries: (case_id, system, snapshot, line, loading, rated)."""
    df = pd.DataFrame(entries, columns=["case_id", "system", "snapshot", "line", "loading", "rated"])
    r = df["rated"].astype(bool)
    df["overload"] = r & (df["loading"] > 1.0)
    df["severe_overload"] = r & (df["loading"] > 1.2)
    po = np.zeros(len(df), bool) if pre_overload is None else np.asarray(pre_overload, bool)
    df["induced_overload"] = df["overload"] & ~po
    if status is not None:
        df["status"] = status
    return df


def _empty_lines():
    return _line_frame([])


def _random_assess(seed=0):
    """3 cases (2 systems) x 4 snapshots x 4 buses / 4 lines (l3 unrated); 'dc' noisy, 'gnn' == truth."""
    rng = np.random.default_rng(seed)
    keys = [(c, s, t) for c, s in ((0, "A"), (1, "A"), (2, "B")) for t in range(4)]
    be = [(c, s, t, f"b{i}") for c, s, t in keys for i in range(4)]
    le = [(c, s, t, f"l{i}") for c, s, t in keys for i in range(4)]
    v_true = rng.uniform(0.80, 1.20, len(be))
    v_dc = v_true + rng.normal(0.0, 0.06, len(be))
    pre_b = rng.random((len(be), 2)) < 0.3
    rated = np.array([e[3] != "l3" for e in le])
    l_true = np.where(rated, rng.uniform(0.5, 1.5, len(le)), np.nan)
    l_dc = np.where(rated, l_true + rng.normal(0.0, 0.15, len(le)), np.nan)
    pre_l = rng.random(len(le)) < 0.3
    bus = lambda v: _bus_frame([(*e, x) for e, x in zip(be, v)], pre=pre_b)
    line = lambda l: _line_frame([(*e, x, r) for e, x, r in zip(le, l, rated)], pre_overload=pre_l)
    dc_line = line(l_dc)
    for c in ("overload", "severe_overload", "induced_overload"):   # junk labels on unrated lines must be ignored
        dc_line.loc[~rated, c] = True
    return {"truth": (bus(v_true), line(l_true)), "dc": (bus(v_dc), dc_line),
            "gnn": (bus(v_true), line(l_true)), "_timing_s": {"truth": 1.0, "dc": 0.5, "gnn": 0.2}}


def _counts(t, p):
    t, p = np.asarray(t, bool), np.asarray(p, bool)
    return dict(tp=int((t & p).sum()), fp=int((~t & p).sum()), fn=int((t & ~p).sum()), tn=int((~t & ~p).sum()))


# ── classification_counts ────────────────────────────────────────────────────
def test_classification_counts(ns):
    c = ns["classification_counts"]([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (c["tp"], c["fp"], c["fn"], c["tn"], c["n"]) == (2, 1, 1, 1, 5)
    for k in ("precision", "recall", "f1"):
        assert c[k] == pytest.approx(2 / 3)
    assert c["miss_rate"] == pytest.approx(1 / 3)
    assert c["specificity"] == pytest.approx(0.5)
    z = ns["classification_counts"]([0, 0, 0], [0, 0, 0])
    assert (z["tp"], z["fp"], z["fn"], z["tn"]) == (0, 0, 0, 3)
    assert np.isnan(z["recall"]) and np.isnan(z["precision"]) and np.isnan(z["f1"]) and np.isnan(z["miss_rate"])


# ── element level ────────────────────────────────────────────────────────────
def test_element_level_metrics(ns):
    a = _random_assess()
    df = ns["element_level_metrics"](a)
    assert set(df.predictor) == {"dc", "gnn"} and set(df.system) == {"ALL", "A", "B"}
    assert len(df) == 2 * 3 * 3 * 3   # predictors x systems x violation types x levels
    for p in ("dc", "gnn"):
        for (el, vt), levels in LABELS.items():
            t, pr = a["truth"][0 if el == "bus" else 1], a[p][0 if el == "bus" else 1]
            keep = t["rated"].to_numpy(bool) if el == "line" else np.ones(len(t), bool)
            for system in ("ALL", "A", "B"):
                sel = keep & ((t.system == system).to_numpy() if system != "ALL" else True)
                for level, col in levels.items():
                    row = df[(df.predictor == p) & (df.system == system) & (df.violation_type == vt)
                             & (df.level == level)]
                    assert len(row) == 1, (p, system, vt, level)
                    want = _counts(t[col].to_numpy()[sel], pr[col].to_numpy()[sel])
                    assert {k: int(row.iloc[0][k]) for k in want} == want, (p, system, vt, level)
                    assert int(row.iloc[0]["n"]) == int(sel.sum())
    lines_all = df[(df.element == "line") & (df.system == "ALL")]
    assert (lines_all["n"] == 36).all()   # 12 case-snapshots x 3 rated lines; unrated l3 excluded
    assert set(lines_all.level) == set(LEVELS)   # incl. thermal contingency_induced (D11b(b))
    assert (df[df.predictor == "gnn"][["fp", "fn"]] == 0).all().all()
    dc = df[df.predictor == "dc"]
    assert (dc[["fp", "fn"]].sum(axis=1) > 0).any()


# ── case level ───────────────────────────────────────────────────────────────
def test_case_snapshot_labels_any_over_elements(ns):
    bus, line = _random_assess()["truth"]
    lab = ns["case_snapshot_labels"](bus, line).set_index(["case_id", "snapshot"])
    rated = line[line.rated]
    for lv, (bc, lc) in {"violation": ("violation", "overload"), "severe": ("severe", "severe_overload"),
                         "contingency_induced": ("contingency_induced", "induced_overload")}.items():
        v = bus.groupby(["case_id", "snapshot"])[bc].any()
        th = rated.groupby(["case_id", "snapshot"])[lc].any()
        assert (lab[f"any_{lv}_voltage"] == v.loc[lab.index]).all(), lv
        assert (lab[f"any_{lv}_thermal"] == th.loc[lab.index]).all(), lv
        assert (lab[f"any_{lv}_any"] == (v | th).loc[lab.index]).all(), lv
    np.testing.assert_allclose(lab["max_loading"], rated.groupby(["case_id", "snapshot"])["loading"].max().loc[lab.index])
    np.testing.assert_allclose(
        lab["sum_overload"],
        rated.assign(o=np.maximum(0, rated.loading - 1)).groupby(["case_id", "snapshot"])["o"].sum().loc[lab.index])
    np.testing.assert_allclose(lab["min_v"], bus.groupby(["case_id", "snapshot"])["vmag"].min().loc[lab.index])
    dev = np.maximum(0, np.maximum(0.9 - bus.vmag, bus.vmag - 1.1))
    np.testing.assert_allclose(lab["max_v_dev"],
                               bus.assign(d=dev).groupby(["case_id", "snapshot"])["d"].max().loc[lab.index])
    assert len(lab) == 12 and set(lab["system"]) == {"A", "B"}


def test_case_level_identical_predictor(ns):
    df = ns["case_level_metrics"](_random_assess())
    assert set(df.scope) == {"voltage", "thermal", "any"}   # no any+no_solution without no_solution_assess
    g = df[df.predictor == "gnn"]
    assert len(g) == 3 * 3 * 3   # systems x levels x scopes
    pos = g[g.tp + g.fn > 0]
    assert len(pos) > 0
    assert (pos.recall == 1.0).all() and (pos.f1 == 1.0).all() and (pos.miss_rate == 0.0).all()
    assert (g.fp == 0).all() and (g.fn == 0).all()


def _no_solution_fixture():
    # solved: 3 case-snapshots (case 0 snapshots 0,1 violate; case 1 snapshot 0 clean); predictor catches both
    t_bus = _bus_frame([(0, "A", 0, "b0", 0.88), (0, "A", 1, "b0", 0.87), (1, "A", 0, "b0", 1.0)])
    p_bus = _bus_frame([(0, "A", 0, "b0", 0.89), (0, "A", 1, "b0", 0.86), (1, "A", 0, "b0", 1.0)])
    assess = {"truth": (t_bus, _empty_lines()), "p": (p_bus, _empty_lines()), "_timing_s": {}}
    # no-solution: 2 outages x 2 snapshots, predictor flags 1 of 4 at level 'violation'
    ns_bus = _bus_frame([(5, "A", 0, "b0", 0.88), (5, "A", 1, "b0", 1.0), (6, "A", 0, "b0", 1.0),
                         (6, "A", 1, "b0", 1.0)], status=["non_converged"] * 2 + ["sanity_rejected"] * 2)
    no_sol = {"p": (ns_bus, _line_frame([], status=[])), "_timing_s": {"p": 0.1}}
    return assess, no_sol


def test_case_level_no_solution(ns):
    assess, no_sol = _no_solution_fixture()
    df = ns["case_level_metrics"](assess, no_sol)
    row = lambda scope, lv="violation": df[(df.predictor == "p") & (df.system == "ALL") & (df.scope == scope)
                                          & (df.level == lv)].iloc[0]
    r = row("any+no_solution")
    assert (r.tp, r.fn, r.fp, r.tn) == (3, 3, 0, 1)
    assert r.recall == pytest.approx(0.5) and r.miss_rate == pytest.approx(0.5)
    solved = row("any")
    assert (solved.tp, solved.fn) == (2, 0) and solved.recall == 1.0
    assert set(df[df.scope == "any+no_solution"].level) == {"violation", "severe"}
    assert set(df[df.scope == "any+no_solution"].system) == {"ALL", "A"}
    sev = row("any+no_solution", "severe")   # nothing severe predicted; 4 no-solution positives missed
    assert (sev.tp, sev.fn) == (0, 4)
    without = ns["case_level_metrics"](assess)
    assert "any+no_solution" not in set(without.scope)
    pd.testing.assert_frame_equal(without, df[df.scope != "any+no_solution"].reset_index(drop=True))
    # predictor absent from no_solution_assess -> n_ns = 0 -> equals the solved-only 'any' row
    absent = ns["case_level_metrics"](assess, {"_timing_s": {}})
    a, s = (absent[(absent.system == "ALL") & (absent.scope == sc) & (absent.level == "violation")].iloc[0]
            for sc in ("any+no_solution", "any"))
    assert (a.tp, a.fp, a.fn, a.tn) == (s.tp, s.fp, s.fn, s.tn)


def test_no_solution_flag_table(ns):
    _, no_sol = _no_solution_fixture()
    ft = ns["no_solution_flag_table"](no_sol)
    r = ft[(ft.predictor == "p") & (ft.system == "ALL") & (ft.status == "ALL") & (ft.level == "violation")].iloc[0]
    assert (r.n_outages, r.n_case_snapshots, r.n_flagged_case_snapshots, r.n_outages_flagged_any_snapshot) == (2, 4, 1, 1)
    assert r.flag_rate_snapshot == pytest.approx(0.25) and r.flag_rate_outage == pytest.approx(0.5)
    nc = ft[(ft.system == "ALL") & (ft.status == "non_converged") & (ft.level == "violation")].iloc[0]
    assert (nc.n_outages, nc.n_case_snapshots, nc.flag_rate_snapshot, nc.flag_rate_outage) == (1, 2, 0.5, 1.0)
    assert set(ft.status) == {"ALL", "non_converged", "sanity_rejected"}
    assert ns["no_solution_flag_table"]({"_timing_s": {}}).empty


# ── outage summary ───────────────────────────────────────────────────────────
def _sidecar(bad_nc=(), bad_san=()):
    st = ["bridge_skipped", "bridge_skipped", "non_converged", "sanity_rejected", "solved", "solved", "solved"]
    return pd.DataFrame(dict(case_id=range(7), status=st, system="A", source_index="src", source_position=0,
                             nonconverged_snapshots=[[], [], ["0", "1", "2"], [], [], [], []],
                             file_position=[None, None, None, None, 0, 1, 2],
                             base_nonconverged_snapshots=[list(bad_nc) for _ in range(7)],   # [STSI 151026]:task 093
                             base_sanity_failed_snapshots=[list(bad_san) for _ in range(7)]))


def test_outage_summary(ns):
    rows = _sidecar()
    bases = {("src", 0): SimpleNamespace(snapshots=pd.RangeIndex(24))}
    v = {4: [0.88] * 24, 5: [0.80] * 12 + [1.0] * 12, 6: [1.0] * 24}   # 24 violation + 12 (also severe)
    bus = _bus_frame([(c, "A", t, "b0", v[c][t]) for c in (4, 5, 6) for t in range(24)])
    labels = ns["case_snapshot_labels"](bus, _empty_lines())
    s = ns["outage_summary"](rows, labels, bases)
    assert list(s.system) == ["ALL", "A"]
    r = s[s.system == "ALL"].iloc[0]
    assert (r.n_outages, r.n_islanding, r.n_non_converged, r.n_sanity_rejected, r.n_solved) == (7, 2, 1, 1, 3)
    assert (r.cs_solved, r.cs_islanding, r.cs_no_solution_severe) == (72, 48, 48)
    assert r.cs_base_nonconverged_no_solution == 0 and r.cs_base_nonconverged_solved == 0   # [STSI 151026]:task 093
    assert r.share_any_violation_solved == pytest.approx(36 / 72)
    assert r.share_any_severe_solved == pytest.approx(12 / 72)
    assert r.share_any_violation_incl_no_solution == pytest.approx((36 + 48) / 120)
    assert r.share_any_severe_incl_no_solution == pytest.approx((12 + 48) / 120)   # bridges never severe
    s_nc = ns["outage_summary"](rows, labels, bases, no_solution_statuses=("non_converged",))
    r = s_nc[s_nc.system == "ALL"].iloc[0]
    assert (r.cs_islanding, r.cs_no_solution_severe) == (48, 24)
    assert r.share_any_severe_incl_no_solution == pytest.approx((12 + 24) / 96)


def test_outage_summary_base_failed(ns):
    # [STSI 151026]:task 093 - snapshots "0","1" base-failed (union, not double-counted); solved labels filtered upstream
    rows = _sidecar(bad_nc=["0"], bad_san=["0", "1"])
    bases = {("src", 0): SimpleNamespace(snapshots=pd.RangeIndex(24))}
    v = {4: [0.88] * 24, 5: [0.80] * 12 + [1.0] * 12, 6: [1.0] * 24}
    bus = _bus_frame([(c, "A", t, "b0", v[c][t]) for c in (4, 5, 6) for t in range(2, 24)])
    labels = ns["case_snapshot_labels"](bus, _empty_lines())
    r = ns["outage_summary"](rows, labels, bases).set_index("system").loc["ALL"]
    assert (r.cs_solved, r.cs_islanding, r.cs_no_solution_severe) == (66, 48, 44)
    assert (r.cs_base_nonconverged_no_solution, r.cs_base_nonconverged_solved) == (4, 6)
    assert r.share_any_violation_incl_no_solution == pytest.approx((32 + 44) / 110)
    assert r.share_any_severe_incl_no_solution == pytest.approx((10 + 44) / 110)
    r = ns["outage_summary"](rows, labels, bases, no_solution_statuses=("non_converged",)).set_index("system").loc["ALL"]
    assert (r.cs_no_solution_severe, r.cs_base_nonconverged_no_solution) == (22, 2)


# ── severity / PR-AUC / near-limit ───────────────────────────────────────────
def _severity_assess(pred_order):
    """6 cases, 1 snapshot, 1 bus + 1 rated line each; distinct severities; pred values permuted by pred_order."""
    v = [0.89, 0.87, 1.14, 1.16, 0.83, 1.20]         # band deviations 0.01, 0.03, 0.04, 0.06, 0.07, 0.10
    l = [0.60, 0.80, 1.00, 1.10, 1.30, 1.50]
    frames = lambda vv, ll: (_bus_frame([(c, "A", 0, "b0", vv[c]) for c in range(6)]),
                             _line_frame([(c, "A", 0, "l0", ll[c], True) for c in range(6)]))
    return {"truth": frames(v, l), "same": frames(v, l),
            "rev": frames([v[i] for i in pred_order], [l[i] for i in pred_order]), "_timing_s": {}}


def test_severity_ranking(ns):
    df = ns["severity_ranking"](_severity_assess([5, 4, 3, 2, 1, 0]), k_list=(2, 3, 10))
    for sev in ("thermal", "voltage"):
        same = df[(df.predictor == "same") & (df.severity == sev)].iloc[0]
        assert same.n_cases == 6 and same.spearman_rho == pytest.approx(1.0)
        assert same.top2_recall == 1.0 and same.top3_recall == 1.0 and np.isnan(same.top10_recall)   # k > n
        rev = df[(df.predictor == "rev") & (df.severity == sev)].iloc[0]
        assert rev.spearman_rho == pytest.approx(-1.0)
        assert rev.top2_recall == 0.0 and rev.top3_recall == 0.0
    assert list(ns["severity_ranking"](_severity_assess(range(6))).columns[-3:]) == \
        ["top5_recall", "top10_recall", "top20_recall"]


def test_pr_auc(ns):
    t_bus = _bus_frame([(0, "A", 0, f"b{i}", v) for i, v in enumerate([0.84, 0.88, 0.95, 1.0])])
    p_bus = _bus_frame([(0, "A", 0, f"b{i}", v) for i, v in enumerate([0.86, 0.89, 0.97, 1.05])])   # same ranking
    t_line = _line_frame([(0, "A", 0, "l0", 1.1, True), (0, "A", 0, "l1", 0.9, True), (0, "A", 0, "l2", np.nan, False)])
    p_line = _line_frame([(0, "A", 0, "l0", 0.95, True), (0, "A", 0, "l1", 0.5, True), (0, "A", 0, "l2", 9.0, False)])
    df = ns["pr_auc_table"]({"truth": (t_bus, t_line), "p": (p_bus, p_line), "_timing_s": {}})
    g = lambda vt, lv: df[(df.violation_type == vt) & (df.level == lv)].iloc[0]
    assert g("under_v", "violation").pr_auc == pytest.approx(1.0) and g("under_v", "violation").n_pos == 2
    assert g("under_v", "severe").pr_auc == pytest.approx(1.0) and g("under_v", "severe").n_pos == 1
    assert np.isnan(g("over_v", "violation").pr_auc) and g("over_v", "violation").n_pos == 0
    assert g("overload", "violation").pr_auc == pytest.approx(1.0) and g("overload", "violation").n == 2   # rated only
    assert np.isnan(g("overload", "severe").pr_auc)


def test_near_limit_mae(ns):
    vt = [0.86, 0.80, 1.00, 1.14, 1.06]                # in, out, out, in, in (tol 5 %)
    dv = [0.01, 0.50, 0.50, -0.02, 0.03]
    lt = [0.97, 1.04, 0.50, 1.20]                     # in, in, out, out
    dl = [0.02, -0.04, 0.30, 0.30]
    t_bus = _bus_frame([(0, "A", 0, f"b{i}", v) for i, v in enumerate(vt)])
    p_bus = _bus_frame([(0, "A", 0, f"b{i}", v + d) for i, (v, d) in enumerate(zip(vt, dv))])
    t_line = _line_frame([(0, "A", 0, f"l{i}", x, True) for i, x in enumerate(lt)] + [(0, "A", 0, "lu", 1.0, False)])
    p_line = _line_frame([(0, "A", 0, f"l{i}", x + d, True) for i, (x, d) in enumerate(zip(lt, dl))]
                         + [(0, "A", 0, "lu", 1.5, False)])
    df = ns["near_limit_mae"]({"truth": (t_bus, t_line), "p": (p_bus, p_line), "_timing_s": {}})
    b = df[df.element == "bus"].iloc[0]
    assert b.n == 3 and b.mae == pytest.approx(0.02) and b.unit == "pu"
    l = df[df.element == "line"].iloc[0]
    assert l.n == 2 and l.mae == pytest.approx(3.0) and l.unit == "pct_points"


# ── cost / regression / save ─────────────────────────────────────────────────
def test_cost_table(ns):
    df = ns["cost_table"]({"g": {"batch_solve_time_ms": 0.5, "total_time_ms": 2.0}},
                          {"PyPower AC": {"inference_time_per_snapshot_ms": 10.0}},
                          {"DC lpf": {"inference_time_per_snapshot_ms": 1.0}},
                          {"truth": 4.8, "dc": 2.4}, n_cases=10, n_snapshots=24)
    assert len(df) == 6
    np.testing.assert_allclose(df.ms_per_contingency, df.ms_per_snapshot * 24)
    m = df.set_index("method")
    assert m.loc["GNN g (batched)", "ms_per_contingency"] == pytest.approx(12.0)
    assert m.loc["GNN g (single)", "ms_per_contingency"] == pytest.approx(48.0)
    assert m.loc["PyPower AC (single)", "ms_per_contingency"] == pytest.approx(240.0)
    assert m.loc["pipeline (incl. pandas): truth", "ms_per_contingency"] == pytest.approx(480.0)
    assert m.loc["pipeline (incl. pandas): dc", "ms_per_snapshot"] == pytest.approx(10.0)


def test_n1_vs_base_regression(ns, monkeypatch):
    calls = []

    def fake_eval(model, nets, **kw):
        calls.append((model, len(nets), kw))
        return {"vmag_mae": 0.01 * len(nets), "p_mae": 0.5, "batch_solve_time_ms": 1.0}

    monkeypatch.setitem(ns, "evaluate_gnn_on_test_set", fake_eval)
    runs = {"r1": {"model": "M", "hparams": {"max_buses": 30, "ptdf_offset": 2}}}
    df, raw = ns["n1_vs_base_regression"](runs, ["n"] * 5, ["b"] * 2)
    assert list(df["set"]) == ["base", "n1"] and list(df.run_key) == ["r1", "r1"]
    assert df.vmag_mae.tolist() == pytest.approx([0.02, 0.05])
    assert np.isnan(df.delta_theta_mae).all() and np.isnan(df.line_flow_q0_mae).all()
    assert set(raw) == {("r1", "base"), ("r1", "n1")}
    assert [c[1] for c in calls] == [2, 5]
    kw = calls[0][2]
    assert (kw["max_buses"], kw["ptdf_offset"], kw["ptdf_params"], kw["use_pnom_share"], kw["verbose"]) == \
        (30, 2, None, None, False)


def test_save_contingency_report(ns, tmp_path):
    r = {"a": pd.DataFrame({"x": [1, 2]}), "b": pd.DataFrame({"y": [3.0]})}
    with pytest.raises(FileExistsError):
        ns["save_contingency_report"](r, str(tmp_path))
    out = tmp_path / "report_new"
    paths = ns["save_contingency_report"](r, str(out))
    assert sorted(os.listdir(out)) == ["a.csv", "b.csv"] and len(paths) == 2
    pd.testing.assert_frame_equal(pd.read_csv(out / "a.csv"), r["a"])


def test_build_and_print_report(ns, capsys):
    assess = _random_assess()
    _, no_sol = _no_solution_fixture()
    no_sol = {"dc": no_sol["p"], "gnn": no_sol["p"], "_timing_s": {}}
    rows = pd.DataFrame(dict(case_id=[0, 1, 2, 5, 6, 7], system=["A", "A", "B", "A", "A", "B"],
                             status=["solved"] * 3 + ["non_converged", "sanity_rejected", "bridge_skipped"],
                             source_index="src", source_position=0,
                             base_nonconverged_snapshots=[[] for _ in range(6)],   # [STSI 151026]:task 093
                             base_sanity_failed_snapshots=[[] for _ in range(6)]))
    bases = {("src", 0): SimpleNamespace(snapshots=pd.RangeIndex(4))}
    r = ns["build_contingency_report"](assess, no_sol, rows, bases, {}, [], {}, {})
    assert set(r) == {"outage_summary", "case_level", "element_level", "no_solution_flags", "severity_ranking",
                      "pr_auc", "near_limit_mae", "regression_n1_vs_base", "cost"}
    assert r["regression_n1_vs_base"].empty
    assert len(r["cost"]) == 3   # pipeline rows only (no runs, no reference timing)
    np.testing.assert_allclose(r["cost"].ms_per_contingency, r["cost"].ms_per_snapshot * 4)
    ns["print_contingency_report"](r)
    out = capsys.readouterr().out
    i1, i2, io = out.index("HEADLINE 1"), out.index("HEADLINE 2"), out.index("OUTAGE SUMMARY")
    assert i1 < i2 < io < out.index("NO-SOLUTION FLAG TABLE")
    assert ns["NO_SOLUTION_CAVEAT"] in out[i1:i2] and "any+no_solution" in out[i1:i2]
    assert out.rstrip().endswith(ns["D3_CAVEAT"])


# [STSI 151026]:task 093 D-C - regression nets restricted to base-ok snapshots
def _tiny_net(ns):
    n = ns["pypsa"].Network()
    n.set_snapshots(range(4))
    n.add("Bus", "b")
    return n


def test_restrict_nets_to_base_ok(ns):
    n = _tiny_net(ns)
    m = n.copy()
    rows = lambda bad: pd.DataFrame(dict(case_id=[0, 1], status=["solved", "bridge_skipped"], source_index="src",
                                         source_position=0, file_position=[0, None],
                                         base_nonconverged_snapshots=[[], []],
                                         base_sanity_failed_snapshots=[list(bad), list(bad)]))
    n1, base = ns["restrict_nets_to_base_ok"]([n], rows([]), {("src", 0): m})
    assert len(n1) == 1 and n1[0] is n and len(base) == 1 and base[0] is m
    n1, base = ns["restrict_nets_to_base_ok"]([n], rows(["1"]), {("src", 0): m})
    assert [list(x.snapshots) for x in (*n1, *base)] == [[0, 2, 3], [0, 2, 3]]
    assert n1[0] is not n and base[0] is not m
    assert len(n.snapshots) == 4 and len(m.snapshots) == 4
    assert ns["restrict_nets_to_base_ok"]([n], rows(["0", "1", "2", "3"]), {("src", 0): m}) == ([], [])


def test_build_report_restricts_regression_nets(ns, monkeypatch):
    seen = {}

    def fake_reg(runs, n1_nets, base_net_list):
        seen["n1"] = [len(n.snapshots) for n in n1_nets]
        seen["base"] = [len(n.snapshots) for n in base_net_list]
        return pd.DataFrame(columns=["run_key", "set", *ns["_REG_COLS"]]), {}

    monkeypatch.setitem(ns, "n1_vs_base_regression", fake_reg)
    tiny4 = _tiny_net(ns)
    rows = pd.DataFrame(dict(case_id=[0, 1, 2], system=["A", "A", "B"], status="solved", source_index="src",
                             source_position=0, file_position=[0, 1, 2],
                             base_nonconverged_snapshots=[[] for _ in range(3)],
                             base_sanity_failed_snapshots=[["3"] for _ in range(3)]))
    ns["build_contingency_report"](_random_assess(), None, rows, {("src", 0): tiny4}, {"r": {}}, [tiny4] * 3, {}, {})
    assert seen == {"n1": [3, 3, 3], "base": [3]}


# ── notebook exec / driver ───────────────────────────────────────────────────
def _driver_src():
    return next(s for s in _code_sources(CONTINGENCY) if DRIVER_BANNER in s)


def test_exec_all_cells_driver_inert():
    fresh = _exec_all("__task090_fresh__")
    assert fresh["run_contingency_eval"] is False
    assert fresh["N1_INDEX_PATH"] is None and fresh["RUN_KEYS"] is None and fresh["SWEEP_PATHS"] is None
    for name in ("assess", "no_sol", "report", "n1_nets", "runs"):
        assert name not in fresh, name


def test_cell_order_and_source_guard():
    srcs = _code_sources(CONTINGENCY)
    i091 = next(i for i, s in enumerate(srcs) if "# ==== CONTINGENCY: NO-SOLUTION OUTAGES (task 091) ====" in s)
    ilib = next(i for i, s in enumerate(srcs) if LIB_BANNER in s)
    idrv = next(i for i, s in enumerate(srcs) if DRIVER_BANNER in s)
    assert i091 < ilib < idrv == len(srcs) - 1
    drv = srcs[idrv]
    assert "NO_SOLUTION_GRANULARITY" not in drv and "NO_SOLUTION_GRANULARITY" not in srcs[ilib]
    assert "run_no_solution_assessment(" in drv and "build_contingency_report(" in drv
    assert 'if globals().get("run_contingency_eval", False):' in drv   # [STSI 141026]:driver reads the 088 toggle, no local copy
    assert "run_contingency_eval = False" not in drv
    assert "RUN_KEYS      = None" in drv and "N1_INDEX_PATH = None" in drv   # D6/D7: supplied at call time
    assert "NO_SOLUTION_STATUSES = " not in srcs[ilib]   # reuses task 091's constant


def test_driver_wiring(ns):
    """Driver with the toggle flipped and every step stubbed: call order and data flow."""
    calls = []

    def stub(name, ret):
        def f(*a, **k):
            calls.append((name, a))
            return ret
        return f

    env = dict(ns)
    env.update(run_contingency_eval=True, run_ref_timing=False,   # [STSI 141026]:flip the 088 toggle in the namespace
               load_n1_dataset=stub("load_n1_dataset", ("NETS", "HDR", "ROWS")),
               load_source_base_nets=stub("load_source_base_nets", "BASES"),
               load_runs_by_key=stub("load_runs_by_key", "RUNS"),
               run_violation_assessment=stub("run_violation_assessment", "ASSESS"),
               run_no_solution_assessment=stub("run_no_solution_assessment", "NOSOL"),
               run_reference_timing=stub("run_reference_timing", ("RS", "RB")),
               build_contingency_report=stub("build_contingency_report", "REPORT"),
               print_contingency_report=stub("print_contingency_report", None),
               save_contingency_report=stub("save_contingency_report", None))
    src = _driver_src().replace("RUN_KEYS      = None", "RUN_KEYS      = ['k']")
    exec(compile(src, "<driver>", "exec"), env)  # noqa: S102
    names = [c[0] for c in calls]
    assert names == ["load_n1_dataset", "load_source_base_nets", "load_runs_by_key", "run_violation_assessment",
                     "run_no_solution_assessment", "build_contingency_report", "print_contingency_report"]
    d = dict(calls)
    assert d["run_no_solution_assessment"] == ("ROWS", "BASES", "HDR", "RUNS")
    assert d["build_contingency_report"] == ("ASSESS", "NOSOL", "ROWS", "BASES", "RUNS", "NETS", {}, {})
    assert d["load_runs_by_key"] == (["k"], None)
