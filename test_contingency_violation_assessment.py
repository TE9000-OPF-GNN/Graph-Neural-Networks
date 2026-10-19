"""Task 089: exec-based tests of the Contingency notebook's violation-assessment cell.

Contingency code cells are exec'd in order in a fresh namespace (IPython magics commented out, toggles cell skipped).
Real-data tests use base net mixed_1500_w position 138 (ieee30) with non-bridge lines removed and re-solved in memory;
they are skipped if the data is missing. Existing dataset files are only read.
"""
# [STSI 141026]:new violation-assessment tests for the contingency notebook (task 089)
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
CONTINGENCY = os.path.join(REPO, "GNN_Powerflow_V2.7_Contingency.ipynb")
TOGGLES_MARKER = "run_contingency_eval = False"
S8_NAME = "sweep_S8_new_curriculum.json"
BASE_NAME = "mixed_1500_w.json"
BASE_POS = 138


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


@pytest.fixture(scope="module")
def ns():
    namespace = {"__name__": "__task089_test__"}
    for i, src in enumerate(_code_sources(CONTINGENCY)):
        if TOGGLES_MARKER in src:
            continue
        exec(compile(_clean(src), f"<contingency:code{i}>", "exec"), namespace)  # noqa: S102
    main = sys.modules["__main__"]
    had, old = hasattr(main, "PowerFlowGNN"), getattr(main, "PowerFlowGNN", None)
    main.PowerFlowGNN = namespace["PowerFlowGNN"]  # unpickling resolves the class from __main__
    yield namespace
    if had:
        main.PowerFlowGNN = old
    else:
        del main.PowerFlowGNN


# ── hand-built labels ─────────────────────────────────────────────────────────
_VMAG = [0.84, 0.88, 0.90, 1.00, 1.10, 1.12, 1.16]
# (under_v, over_v, violation, severe_under, severe_over, severe) with default bands
_EXPECTED_BUS = [
    (True, False, True, True, False, True),
    (True, False, True, False, False, False),
    (False, False, False, False, False, False),
    (False, False, False, False, False, False),
    (False, False, False, False, False, False),
    (False, True, True, False, False, False),
    (False, True, True, False, True, True),
]
_BUS_COLS = ["under_v", "over_v", "violation", "severe_under", "severe_over", "severe"]


def _one_snapshot(values, prefix):
    cols = [f"{prefix}{i}" for i in range(len(values))]
    return pd.DataFrame([values], index=pd.Index([0], name="snapshot"), columns=cols)


def _bus(ns, post, pre, **kw):
    v, p = _one_snapshot(post, "b"), _one_snapshot(pre, "b")
    return ns["assess_bus_violations"](v, p, pd.Series("PQ", index=v.columns), **kw)


def test_bus_labels_default_bands(ns):
    assert ns["V_BAND_VIOLATION"] == (0.90, 1.10) and ns["V_BAND_SEVERE"] == (0.85, 1.15)
    df = _bus(ns, _VMAG, [1.0] * len(_VMAG))
    assert list(df["bus"]) == [f"b{i}" for i in range(len(_VMAG))]
    assert [tuple(bool(x) for x in r) for r in df[_BUS_COLS].to_numpy()] == _EXPECTED_BUS
    np.testing.assert_allclose(df["margin_v"], np.minimum(np.array(_VMAG) - 0.9, 1.1 - np.array(_VMAG)))
    assert (df["contingency_induced"] == df["violation"]).all()   # pre 1.0 everywhere
    assert set(df["bus_type"]) == {"PQ"}


def test_bus_contingency_induced(ns):
    df = _bus(ns, [0.88, 0.88, 0.95], [0.88, 0.95, 1.12])
    r0, r1, r2 = (df.iloc[i] for i in range(3))
    assert r0.violation and r0.under_v and r0.pre_under and not r0.induced_under and not r0.contingency_induced
    assert r1.induced_under and r1.contingency_induced and not r1.pre_under
    assert r2.pre_over and not (r2.under_v or r2.over_v or r2.violation or r2.severe
                                or r2.induced_under or r2.induced_over or r2.contingency_induced)


def test_bus_custom_bands(ns):
    df = _bus(ns, [0.94, 1.06, 0.88], [1.0, 1.0, 1.0], v_band=(0.95, 1.05), v_band_severe=(0.89, 1.2))
    assert list(df["under_v"]) == [True, False, True]
    assert list(df["over_v"]) == [False, True, False]
    assert list(df["severe_under"]) == [False, False, True]
    assert list(df["contingency_induced"]) == [True, True, True]
    assert not _bus(ns, [0.94, 1.06], [1.0, 1.0])["violation"].any()   # same values, default bands


def _flows(s0, s1, prefix="l"):
    """State dict with |S0| = s0 (all P), |S1| = s1 (all P, opposite sign)."""
    return dict(p0=_one_snapshot(list(s0), prefix), q0=_one_snapshot([0.0] * len(s0), prefix),
                p1=_one_snapshot([-x for x in s1], prefix), q1=_one_snapshot([0.0] * len(s1), prefix))


def test_line_labels(ns):
    # a: end-1 overload; b: severe; c: NaN limit; d/e/f: induced (D11b); g: hypot
    post = _flows([0.9, 1.25, 2.0, 1.10, 1.10, 0.90, 0.6], [1.05, 1.1, 2.0, 1.10, 1.10, 0.90, 0.0])
    post["q0"].iloc[0, 6] = 0.8
    pre = _flows([0.5, 0.5, 2.0, 0.50, 0.90, 1.10, 0.0], [0.5, 0.5, 2.0, 1.05, 0.90, 0.20, 0.0])
    cols = post["p0"].columns
    lim = pd.Series([1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0], index=cols)
    src = pd.Series(["real"] * 2 + ["synthetic"] + ["real"] * 4, index=cols)
    df = ns["assess_line_violations"](post, pre, lim, src).set_index("line")
    a, b, c, d, e, f, g = (df.loc[f"l{i}"] for i in range(7))
    assert a.overload and not a.severe_overload and a.loading == pytest.approx(1.05)
    assert a.loading0 == pytest.approx(0.9) and a.loading1 == pytest.approx(1.05)
    assert b.overload and b.severe_overload and b.loading == pytest.approx(1.25)
    assert not c.rated and c.limit_source == "synthetic"
    assert not (c.overload or c.severe_overload or c.pre_overload or c.induced_overload)
    assert d.overload and d.pre_overload and not d.induced_overload
    assert d.pre_loading == pytest.approx(1.05)   # max(|S0|=0.5, |S1|=1.05)
    assert e.overload and not e.pre_overload and e.induced_overload
    assert f.pre_overload and not (f.overload or f.severe_overload or f.induced_overload)
    assert g.s0 == pytest.approx(1.0) and g.s1 == pytest.approx(0.0)
    rated = df[df.rated]
    np.testing.assert_array_equal(rated["loading"], np.maximum(rated["loading0"], rated["loading1"]))
    assert df["rated"].tolist() == [True, True, False, True, True, True, True]


# ── real data: ieee30 base net with non-bridge outages ────────────────────────
def _non_bridge_lines(net):
    import networkx as nx
    out = []
    for name in net.lines.index:
        G = nx.MultiGraph()
        G.add_nodes_from(net.buses.index)
        for n, r in net.lines.iterrows():
            if n != name:
                G.add_edge(r.bus0, r.bus1)
        for _, r in net.transformers.iterrows():
            G.add_edge(r.bus0, r.bus1)
        if nx.is_connected(G):
            out.append(name)
    return out


@pytest.fixture(scope="module")
def real(ns):
    src = os.path.join(ns["TRAINING_NETWORKS_DIR"], BASE_NAME)
    if not os.path.isfile(src):
        pytest.skip(f"dataset not found: {src}")
    src = os.path.abspath(src)
    (_, _, base), = ns["load_dataset_subset"](src, [BASE_POS])
    lines = _non_bridge_lines(base)[:2]
    assert len(lines) == 2
    posts, rows = [], []
    for i, name in enumerate(lines):
        net = base.copy()
        net.remove("Line", name)
        res = net.pf(use_seed=True)
        assert bool(res.converged.all().all()), name
        net.lines["s_limit_pu"] = 1.0
        net.lines["limit_source"] = "pytest"
        posts.append(net)
        rows.append(dict(case_id=i, status="solved", file_position=i, source_index=src,
                         source_position=BASE_POS, system="ieee30", outage_name=name,
                         base_nonconverged_snapshots=[], base_sanity_failed_snapshots=[]))   # [STSI 151026]:task 093 fields
    rows.append(dict(case_id=2, status="bridge_skipped", file_position=None, source_index=src,
                     source_position=BASE_POS, system="ieee30", outage_name="__bridge__",
                     base_nonconverged_snapshots=[], base_sanity_failed_snapshots=[]))
    return dict(base=base, posts=posts, rows=pd.DataFrame(rows), base_nets={(src, BASE_POS): base}, src=src)


@pytest.fixture(scope="module")
def s8_run(ns):
    p = os.path.join(ns["TRAINING_RESULTS_DIR"], S8_NAME)
    if not os.path.isfile(p):
        pytest.skip(f"sweep file not found: {p}")
    return ns["load_sweep_results"](p)[0]


def test_truth_flow_parity(ns, real):
    net = real["posts"][0]
    n_lines = len(net.lines)
    ds = ns["PowerFlowDataset"]([net], angle_mode="edge_delta", ptdf_branch_mode="all")
    buses, lines = net.buses.index, net.lines.index
    worst = 0.0
    for t_idx, snap in enumerate(net.snapshots):
        data = ds[t_idx]
        ei = data.edge_index[:, ::2].cpu().numpy()
        ea = data.edge_attr[::2].cpu().numpy().astype(np.float64)
        vmag = net.buses_t.v_mag_pu.loc[snap, buses].to_numpy(dtype=float)
        vang = net.buses_t.v_ang.loc[snap, buses].to_numpy(dtype=float)
        fl = ns["calculate_line_flows_from_delta_theta"](vang[ei[0]] - vang[ei[1]], vmag, ei, ea)
        p0 = net.lines_t.p0.loc[snap, lines].to_numpy(dtype=float)
        q0 = net.lines_t.q0.loc[snap, lines].to_numpy(dtype=float)
        s1 = np.hypot(net.lines_t.p1.loc[snap, lines].to_numpy(dtype=float),
                      net.lines_t.q1.loc[snap, lines].to_numpy(dtype=float))
        worst = max(worst, np.abs(fl["p0"][:n_lines] - p0).max(), np.abs(fl["q0"][:n_lines] - q0).max(),
                    np.abs(np.hypot(fl["p1"][:n_lines], fl["q1"][:n_lines]) - s1).max())
    assert worst < 1e-6, worst


def test_predict_state_gnn(ns, real, s8_run):
    net = real["posts"][0]
    st = ns["predict_state_gnn"](s8_run, net)
    assert st["vmag"].shape == (24, len(net.buses))
    assert np.isfinite(st["vmag"].to_numpy()).all()
    for f in ("p0", "q0", "p1", "q1"):
        assert list(st[f].columns) == list(net.lines.index)
        assert np.isfinite(st[f].to_numpy()).all()


def test_predict_state_dc(ns, real):
    net = real["posts"][0]
    st = ns["predict_state_dc"](net)
    bt = ns["bus_types"](net)
    pq = bt.index[bt == "PQ"]
    assert len(pq) > 0
    assert (st["vmag"][pq].to_numpy() == 1.0).all()
    df = ns["assess_bus_violations"](st["vmag"], net.buses_t.v_mag_pu, bt)
    assert not df.loc[df.bus_type == "PQ", "under_v"].any()


def test_run_violation_assessment(ns, real, s8_run):
    key = s8_run["run_key"]
    res = ns["run_violation_assessment"](real["posts"], real["rows"], real["base_nets"], {key: s8_run})
    assert {k for k in res if not k.startswith("_")} == {"truth", "dc", key}
    assert set(res["_timing_s"]) == {"truth", "dc", key}
    n_bus = sum(len(n.snapshots) * len(n.buses) for n in real["posts"])
    n_line = sum(len(n.snapshots) * len(n.lines) for n in real["posts"])
    keysets = {}
    for p in ("truth", "dc", key):
        bus_df, line_df = res[p]
        assert len(bus_df) == n_bus and len(line_df) == n_line
        assert set(bus_df.predictor) == {p} and set(line_df.predictor) == {p}
        assert set(bus_df.case_id) == {0, 1}   # bridge_skipped row ignored
        keysets[p] = (set(map(tuple, bus_df[["case_id", "snapshot", "bus"]].to_numpy())),
                      set(map(tuple, line_df[["case_id", "snapshot", "line"]].to_numpy())))
        assert (bus_df.source_index == real["src"]).all() and (line_df.source_position == BASE_POS).all()
    assert keysets["truth"] == keysets["dc"] == keysets[key]
    assert len(keysets["truth"][0]) == n_bus and len(keysets["truth"][1]) == n_line


def test_run_violation_assessment_drops_base_failed(ns, real):
    # [STSI 151026]:task 093 D-C - base-failed snapshots never scored, for every predictor
    rows = real["rows"].copy()
    s0 = str(real["base"].snapshots[0])
    rows["base_sanity_failed_snapshots"] = [[s0] for _ in range(len(rows))]
    res = ns["run_violation_assessment"](real["posts"], rows, real["base_nets"], {})
    n_bus = sum((len(n.snapshots) - 1) * len(n.buses) for n in real["posts"])
    n_line = sum((len(n.snapshots) - 1) * len(n.lines) for n in real["posts"])
    keysets = {}
    for p in ("truth", "dc"):
        bus_df, line_df = res[p]
        assert len(bus_df) == n_bus and len(line_df) == n_line, p
        assert s0 not in set(bus_df.snapshot.map(str)) and s0 not in set(line_df.snapshot.map(str)), p
        keysets[p] = (set(map(tuple, bus_df[["case_id", "snapshot", "bus"]].to_numpy())),
                      set(map(tuple, line_df[["case_id", "snapshot", "line"]].to_numpy())))
    assert keysets["truth"] == keysets["dc"]


def test_run_reference_timing(ns, real):
    single, batch = ns["run_reference_timing"]([real["posts"][0]], n_subset=1)
    merged = {**single, **batch}
    for k in ("PyPower AC", "FDLF", "pandapower", "PyPSA single", "PyPSA batch", "DC lpf"):
        assert k in merged, k
        assert "inference_time_per_snapshot_ms" in merged[k], k
