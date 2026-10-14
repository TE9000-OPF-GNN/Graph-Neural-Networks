"""Task 086: N-1 DataGen foundation — solve_power_flow_with_slack returns pf() result + per-line limits.

Group 0: notebook-source guards (D10 edit, new limits cell present).
Group 1: functional tests on notebook-extracted functions (pf return, identify_base_system,
         load_real_line_ratings, compute_synthetic_line_limits, attach_line_limits).
"""
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.6_DataGen.ipynb")

# first code cell containing each marker is exec'd, in this order
_CELL_MARKERS = [
    "import pypsa",
    "TRAINING_NETWORKS_DIR =",
    "def solve_power_flow_with_slack(",
    "def add_ieee9_branches(",
    "def save_dataset_list(",
    "N-1 CONTINGENCY: LIMITS (task 086)",
]


def _load_notebook_cells():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _cell_source(marker):
    for cell in _load_notebook_cells():
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if marker in src:
                return src
    raise AssertionError(f"cell with marker {marker!r} not found")


@pytest.fixture(scope="module")
def ns():
    namespace = {"__name__": "__task086_test__"}
    for marker in _CELL_MARKERS:
        exec(compile(_cell_source(marker), f"<notebook:{marker}>", "exec"), namespace)  # noqa: S102
    return namespace


def _grid_dir(ns):
    path = ns["GRID_MODEL_FILES"]
    if not os.path.isdir(path):
        pytest.skip(f"GRID_MODEL_FILES not found: {path}")
    return path


# ── Group 0: source guards ───────────────────────────────────────────────────
def test_solve_pf_returns_pf_result_source():
    src = _cell_source("def solve_power_flow_with_slack(")
    assert src.count("return network.pf(") == 3
    bare = [ln for ln in src.splitlines()
            if ln.strip() in ("network.pf(use_seed=True)", "network.pf(use_seed=True, distribute_slack=True)")]
    assert bare == []


def test_limits_cell_present_and_never_writes_s_nom():
    cells = _load_notebook_cells()
    assert len(cells) >= 75  # [STSI 141026]:task 087 appends cells after 086's; guard only that 086's cells exist
    assert cells[73]["cell_type"] == "markdown"
    assert "".join(cells[73]["source"]).startswith("## N-1 Contingency Dataset Generation")
    src = "".join(cells[74]["source"])
    assert src.startswith("# ==== N-1 CONTINGENCY: LIMITS (task 086) ====")
    for forbidden in ('["s_nom"] =', '["v_nom"] =', '["tap_ratio"] =', "sn_mva ="):
        assert forbidden not in src


# ── Group 1: functional ──────────────────────────────────────────────────────
def _tiny_net(pypsa):
    net = pypsa.Network()
    net.set_snapshots([0])
    for k in (1, 2, 3):
        net.add("Bus", f"Bus {k}", v_nom=1.0)
    net.add("Line", "L12", bus0="Bus 1", bus1="Bus 2", r=0.01, x=0.1, s_nom=1.0)
    net.add("Line", "L23", bus0="Bus 2", bus1="Bus 3", r=0.01, x=0.1, s_nom=1.0)
    net.add("Generator", "G1", bus="Bus 1", control="Slack", p_nom=2.0, p_set=0.2)
    net.add("Generator", "G2", bus="Bus 2", control="PV", p_nom=1.0, p_set=0.3)
    net.add("Load", "D3", bus="Bus 3", p_set=0.5, q_set=0.1)
    return net


@pytest.mark.parametrize("distribute", [False, True])
def test_solve_pf_returns_converged_dict(ns, distribute):
    net = _tiny_net(ns["pypsa"])
    res = ns["solve_power_flow_with_slack"](net, [], distribute)
    assert res is not None
    assert hasattr(res, "n_iter") and hasattr(res, "error")
    assert bool(res.converged.all().all())


def _stub(names):
    return types.SimpleNamespace(buses=pd.DataFrame(index=pd.Index(names)))


def test_identify_base_system_synthetic(ns):
    ident = ns["identify_base_system"]
    assert ident(_stub([f"Bus {k}" for k in range(1, 12)])) == "ieee9"
    assert ident(_stub([f"B{k}" for k in range(15)])) == "cigre14"
    assert ident(_stub([f"B{k}" for k in range(15)] + ["Bus 16"])) == "cigre14"
    assert ident(_stub([f"VL{k}_0" for k in range(1, 31)] + ["Bus 31", "Bus 32"])) == "ieee30"
    assert ident(_stub([f"VL{k}_0" for k in range(1, 40)])) == "ieee39"
    with pytest.raises(ValueError):
        ident(_stub([f"VL{k}_0" for k in range(1, 21)]))


def test_identify_base_system_real_data(ns):
    index_path = os.path.join(ns["TRAINING_NETWORKS_DIR"], "mixed_1500_w.json")
    if not os.path.isfile(index_path):
        pytest.skip(f"dataset not found: {index_path}")
    with open(index_path, "r") as fh:
        files = json.load(fh)["files"]
    base = os.path.dirname(index_path)
    for pos, expected in ((120, "ieee9"), (468, "cigre14"), (138, "ieee30")):
        net = ns["pypsa"].Network(os.path.join(base, files[pos]))
        assert ns["identify_base_system"](net) == expected, pos


def test_real_ratings_ieee9(ns):
    r = ns["load_real_line_ratings"]("ieee9")
    expected = {"Line 2": 1.0, "Line 3": 0.75, "Line 5": 1.0, "Line 6": 1.0, "Line 8": 1.0, "Line 9": 1.0}
    assert {k: v for k, (v, _) in r.items()} == expected
    assert {s for _, s in r.values()} == {"code"}


def test_real_ratings_ieee30(ns):
    r = ns["load_real_line_ratings"]("ieee30", grid_model_dir=_grid_dir(ns))
    assert len(r) == 20
    assert {s for _, s in r.values()} == {"raw"}
    assert r["L-1-2-1 "][0] == pytest.approx(1.30)


def test_real_ratings_ieee39(ns):
    r = ns["load_real_line_ratings"]("ieee39", grid_model_dir=_grid_dir(ns))
    assert len(r) == 34
    assert {s for _, s in r.values()} == {"pandapower"}
    vals = np.array([v for v, _ in r.values()])
    assert vals.min() >= 4.8 - 1e-9 and vals.max() <= 12.0 + 1e-9


def test_real_ratings_cigre14(ns):
    r = ns["load_real_line_ratings"]("cigre14", grid_model_dir=_grid_dir(ns))
    assert len(r) == 15
    assert "L-1-2-0" in r
    assert {s for _, s in r.values()} == {"pandapower"}
    vals = np.array([v for v, _ in r.values()])
    assert vals.min() >= 0.05 and vals.max() <= 0.07


@pytest.mark.parametrize("system", ["ieee57", "ieee118"])
def test_real_ratings_empty(ns, system):
    assert ns["load_real_line_ratings"](system) == {}


def _flow_net(pypsa, p0, q0, p1, q1, names=("A", "B")):
    """Net with lines_t flows set directly (rows = snapshots, cols = line names)."""
    net = pypsa.Network()
    net.set_snapshots(range(len(p0)))
    net.add("Bus", "x0")
    net.add("Bus", "x1")
    for n in names:
        net.add("Line", n, bus0="x0", bus1="x1", r=0.01, x=0.1)
    for attr, arr in (("p0", p0), ("q0", q0), ("p1", p1), ("q1", q1)):
        setattr(net.lines_t, attr, pd.DataFrame(np.asarray(arr, dtype=float), index=net.snapshots, columns=list(names)))
    return net


@pytest.mark.parametrize("alpha", [1.2, 1.5])
def test_synthetic_limits_exact_pool_and_nan(ns, alpha):
    pypsa = ns["pypsa"]
    # net1: line A |S0| peaks 5 (3,4), |S1| peaks 1; line B all zero
    n1 = _flow_net(pypsa,
                   p0=[[3.0, 0.0], [0.6, 0.0]], q0=[[4.0, 0.0], [0.8, 0.0]],
                   p1=[[-1.0, 0.0], [0.0, 0.0]], q1=[[0.0, 0.0], [0.0, 0.0]])
    # net2 (same system): line A |S1| peaks 13 (5,12) -> pooled max 13
    n2 = _flow_net(pypsa,
                   p0=[[1.0, 0.0]], q0=[[0.0, 0.0]],
                   p1=[[-5.0, 0.0]], q1=[[-12.0, 0.0]])
    # net3 (other system): line A peaks 2 -> separate key
    n3 = _flow_net(pypsa, p0=[[2.0, 0.0]], q0=[[0.0, 0.0]], p1=[[0.0, 0.0]], q1=[[0.0, 0.0]])
    out = ns["compute_synthetic_line_limits"]([("ieee57", n1), ("ieee57", n2), ("ieee118", n3)], alpha=alpha)
    assert out[("ieee57", "A")] == alpha * 13.0
    assert out[("ieee118", "A")] == alpha * 2.0
    assert np.isnan(out[("ieee57", "B")])
    assert np.isnan(out[("ieee118", "B")])
    # single net only: |S0|=5 > |S1|=1
    single = ns["compute_synthetic_line_limits"]([("ieee57", n1)], alpha=alpha)
    assert single[("ieee57", "A")] == alpha * 5.0


def test_attach_line_limits_preserves_and_roundtrips(ns, tmp_path):
    pypsa = ns["pypsa"]
    net = pypsa.Network()
    net.set_snapshots([0])
    for k in (1, 2, 3):
        net.add("Bus", f"Bus {k}", v_nom=1.0)
    net.add("Line", "Line 2", bus0="Bus 1", bus1="Bus 2", r=0.01, x=0.1, s_nom=1.0)
    net.add("Line", "Line_extra_0", bus0="Bus 2", bus1="Bus 3", r=0.01, x=0.1, s_nom=0.37)
    net.add("Line", "Line_unrated", bus0="Bus 1", bus1="Bus 3", r=0.01, x=0.1, s_nom=1.0)
    net.add("Transformer", "T1", bus0="Bus 1", bus1="Bus 3", x=0.06, s_nom=1.5, tap_ratio=1.0)

    real = {"Line 2": (0.9, "code")}
    synth = {("ieee9", "Line 2"): 5.0, ("ieee9", "Line_extra_0"): 0.42}
    cp = net.copy()
    cp.sn_mva = 100.0  # pypsa 0.28 has no built-in sn_mva; load_system_from_csv sets it as a plain attribute
    # baseline taken from the copy: Network.copy() itself casts build_year int64 -> int32
    lines0, buses0, trafos0 = cp.lines.copy(), cp.buses.copy(), cp.transformers.copy()
    out = ns["attach_line_limits"](cp, "ieee9", real, synth)

    pd.testing.assert_frame_equal(out.lines[lines0.columns], lines0)
    pd.testing.assert_frame_equal(out.buses[buses0.columns], buses0)
    pd.testing.assert_frame_equal(out.transformers[trafos0.columns], trafos0)
    assert out.sn_mva == 100.0
    assert "s_limit_pu" not in net.lines.columns  # original untouched when a copy is passed

    assert out.lines.loc["Line 2", "s_limit_pu"] == 0.9            # real beats synthetic
    assert out.lines.loc["Line 2", "limit_source"] == "code"
    assert out.lines.loc["Line_extra_0", "s_limit_pu"] == 0.42
    assert out.lines.loc["Line_extra_0", "limit_source"] == "synthetic"
    assert np.isnan(out.lines.loc["Line_unrated", "s_limit_pu"])
    assert out.lines.loc["Line_unrated", "limit_source"] == "synthetic"

    path = str(tmp_path / "limits.nc")
    out.export_to_netcdf(path)
    back = pypsa.Network(path)
    assert {"s_limit_pu", "limit_source"} <= set(back.lines.columns)
    assert back.lines.loc["Line 2", "s_limit_pu"] == pytest.approx(0.9)
    assert back.lines.loc["Line_extra_0", "limit_source"] == "synthetic"
    assert back.lines.loc["Line 2", "limit_source"] == "code"
    pd.testing.assert_series_equal(back.lines["s_nom"], out.lines["s_nom"], check_names=False)
