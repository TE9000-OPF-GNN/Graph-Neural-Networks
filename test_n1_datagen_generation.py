"""Task 087: N-1 DataGen contingency generation (test-split base nets -> N-1 line outages -> NEW dataset + sidecar).

Group 0: notebook-source guards (2 appended cells, verbatim reconstruct_test_split copy, driver off).
Group 1: data-independent functional tests (split positions, n_base validation, bridges, sidecar round-trip,
         existence guard).
Group 2: real-data tests (skipped if DATA_ROOT missing). ALL writes go to pytest tmp dirs; the source index is
         only read and its sha256 is checked at the end of the module.
"""
import ast
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(REPO, "GNN_Powerflow_V2.6_DataGen.ipynb")
ANALYSIS = os.path.join(REPO, "GNN_Powerflow_V2.7_Analysis.ipynb")
GEN_MARKER = "# ==== N-1 CONTINGENCY: GENERATION (task 087) ===="
DRV_MARKER = "# ==== N-1 CONTINGENCY: DRIVER (task 087) ===="
COPY_MARKER = "# [COPY of GNN_Powerflow_V2.7_Analysis::reconstruct_test_split]"

# first code cell containing each marker is exec'd, in this order (task 086 list + the two new cells)
_CELL_MARKERS = [
    "import pypsa",
    "TRAINING_NETWORKS_DIR =",
    "def solve_power_flow_with_slack(",
    "def add_ieee9_branches(",
    "def save_dataset_list(",
    "N-1 CONTINGENCY: LIMITS (task 086)",
    GEN_MARKER,
    DRV_MARKER,
]


def _load_cells(path=NOTEBOOK):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _cell_source(marker):
    for cell in _load_cells():
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if marker in src:
                return src
    raise AssertionError(f"cell with marker {marker!r} not found")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_index_path():
    ns = {}
    exec(compile(_cell_source("TRAINING_NETWORKS_DIR ="), "<paths>", "exec"), ns)  # noqa: S102
    return os.path.join(ns["TRAINING_NETWORKS_DIR"], "mixed_1500_w.json")


SOURCE_INDEX = _source_index_path()
# recorded at collection time, i.e. before any test runs
SOURCE_SHA_BEFORE = _sha256(SOURCE_INDEX) if os.path.isfile(SOURCE_INDEX) else None


@pytest.fixture(scope="module")
def ns():
    namespace = {"__name__": "__task087_test__"}
    for marker in _CELL_MARKERS:
        exec(compile(_cell_source(marker), f"<notebook:{marker}>", "exec"), namespace)  # noqa: S102
    return namespace


def _need_source():
    if SOURCE_SHA_BEFORE is None:
        pytest.skip(f"dataset not found: {SOURCE_INDEX}")


def _func_src(src, name):
    segs = [ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(segs) == 1, (name, len(segs))
    return segs[0]


def _same(a, b):
    """Equality treating None/NaN as equal (DataFrame turns None into NaN in numeric columns)."""
    def _missing(x):
        return x is None or (isinstance(x, float) and math.isnan(x))
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return a == b


# ── Group 0: source guards ───────────────────────────────────────────────────
def test_two_cells_appended_after_limits_cell():
    cells = _load_cells()
    assert len(cells) == 77
    assert "".join(cells[74]["source"]).startswith("# ==== N-1 CONTINGENCY: LIMITS (task 086) ====")
    gen = "".join(cells[75]["source"])
    drv = "".join(cells[76]["source"])
    assert cells[75]["cell_type"] == cells[76]["cell_type"] == "code"
    assert gen.startswith(GEN_MARKER)
    assert drv.startswith(DRV_MARKER)
    assert "run_contingency_gen = False" in drv
    assert "N1_N_BASE         = None" in drv
    assert '"mixed_1500_w.json"' in drv
    for src in (gen, drv):
        for forbidden in ('["s_nom"] =', '["v_nom"] =', '["tap_ratio"] =', "sn_mva =", "code_base"):
            assert forbidden not in src


def test_reconstruct_test_split_is_verbatim_copy():
    gen = _cell_source(GEN_MARKER)
    lines = gen.splitlines()
    i = lines.index(COPY_MARKER)
    assert lines[i + 1].startswith("def reconstruct_test_split(")
    an_srcs = [_func_src("".join(c["source"]), "reconstruct_test_split") for c in _load_cells(ANALYSIS)
               if c["cell_type"] == "code" and "def reconstruct_test_split" in "".join(c["source"])]
    assert len(an_srcs) == 1
    assert _func_src(gen, "reconstruct_test_split") == an_srcs[0]


# ── Group 1: data-independent functional ─────────────────────────────────────
def test_split_positions_mixed_1500(ns):
    pos = ns["test_split_positions"](1500)
    assert len(pos) == 225
    assert pos[:8] == [138, 468, 120, 145, 14, 1471, 1205, 471]
    assert pos == list(ns["reconstruct_test_split"](list(range(1500)))[2])


def _fake_index(path, n, tag="fake"):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "timestamp": "x", "files": [f"missing/f{i}.nc" for i in range(n)]}, fh)
    return str(path)


@pytest.mark.parametrize("n_base", [None, 0, -1, 1.5, 226])
def test_select_base_cases_rejects_bad_n_base(ns, tmp_path, n_base):
    idx = _fake_index(tmp_path / "fake.json", 1500)
    with pytest.raises(ValueError):
        ns["select_base_cases"](idx, n_base)


def test_enumerate_n1_line_outages_bridges(ns):
    net = ns["pypsa"].Network()
    for b in ("A", "B", "C", "D", "E", "T"):
        net.add("Bus", b)
    for name, b0, b1 in (("AB", "A", "B"), ("BC", "B", "C"), ("CA", "C", "A"),   # triangle
                         ("CD", "C", "D"),                                        # radial line -> bridge
                         ("DE_1", "D", "E"), ("DE_2", "D", "E")):                 # parallel circuits
        net.add("Line", name, bus0=b0, bus1=b1, x=0.1, r=0.01)
    net.add("Transformer", "TR", bus0="A", bus1="T", x=0.1, s_nom=1.0)           # radial transformer
    out = ns["enumerate_n1_line_outages"](net)
    assert [o["name"] for o in out] == list(net.lines.index)
    assert {o["component"] for o in out} == {"Line"}
    assert {o["name"]: o["is_bridge"] for o in out} == {
        "AB": False, "BC": False, "CA": False, "CD": True, "DE_1": False, "DE_2": False}


def test_sidecar_roundtrip(ns, tmp_path):
    header = dict(tag="n1_x_y", created="2026-10-14T12:00:00", source_index="s.json", source_tag="x",
                  external_index=None, n_base=1, split=dict(seed=42, train_frac=0.7, val_frac=0.15), alpha=1.2,
                  pf_acceptance=dict(ns["N1_PF_ACCEPTANCE"]), dispatch="single_slack_no_qlimits",
                  line_limits={"real": {"ieee9": {"Line 2": [1.0, "code"]}},
                               "synthetic": [["ieee57", "A", 1.5], ["ieee57", "B", float("nan")]]})
    base = dict(source_index="s.json", source_position=3, system="ieee30", outage_component="Line",
                n_buses=33, n_lines_post=40, n_lines_real_rated=19, n_lines_synthetic=21,
                base_nonconverged_snapshots=[], base_sanity_failed_snapshots=["3"])   # [STSI 151026]: task 092 fields
    rows = [
        dict(base, case_id=0, status="solved", file_position=0, outage_name="L1", is_bridge=False, converged=True,
             n_iter=3, pf_error=1e-9, nonconverged_snapshots=[], sanity_reason=None),
        dict(base, case_id=1, status="bridge_skipped", file_position=None, outage_name="L2", is_bridge=True,
             converged=None, n_iter=None, pf_error=None, nonconverged_snapshots=[], sanity_reason=None),
        dict(base, case_id=2, status="non_converged", file_position=None, outage_name="L3", is_bridge=False,
             converged=False, n_iter=100, pf_error=1e18, nonconverged_snapshots=["0", "5"], sanity_reason=None,
             base_nonconverged_snapshots=["0"], base_sanity_failed_snapshots=["0", "5"]),
        dict(base, case_id=3, status="sanity_rejected", file_position=None, outage_name="L4", is_bridge=False,
             converged=True, n_iter=4, pf_error=1e-8, nonconverged_snapshots=[], sanity_reason="ieee30: Voltages"),
    ]
    meta = ns["write_n1_sidecar"](header, rows, str(tmp_path / "m_meta.json"))
    h2, df = ns["read_n1_sidecar"](meta)
    assert "rows" not in h2
    assert set(h2) == set(header)
    for k in header:
        if k != "line_limits":
            assert h2[k] == header[k], k
    assert h2["line_limits"]["real"] == header["line_limits"]["real"]
    syn = h2["line_limits"]["synthetic"]
    assert syn[0] == ["ieee57", "A", 1.5]
    assert syn[1][:2] == ["ieee57", "B"] and math.isnan(syn[1][2])
    assert len(df) == len(rows)
    for r, r2 in zip(rows, df.to_dict(orient="records")):
        assert set(r2) == set(r)
        for k in r:
            assert _same(r[k], r2[k]), (k, r[k], r2[k])


def test_existence_guard_before_loading(ns, tmp_path):
    src = _fake_index(tmp_path / "src.json", 1500, tag="fakesrc")   # .nc files do not exist
    out = tmp_path / "out"
    out.mkdir()
    (out / "n1_fakesrc_d_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ns["generate_n1_contingency_dataset"](src, 1, "d", out_dir=str(out))
    for bad in (None, "", "a b", "a/b", 5):
        with pytest.raises(ValueError):
            ns["generate_n1_contingency_dataset"](src, 1, bad, out_dir=str(out))
    assert sorted(p.name for p in out.iterdir()) == ["n1_fakesrc_d_meta.json"]


# ── Group 2: real data (read-only source; all writes in tmp dirs) ─────────────
def _load_position(ns, pos):
    return ns["load_dataset_subset"](SOURCE_INDEX, [pos])[0][2]


def test_solve_n1_case_ieee9_all_nonbridge_lines(ns):
    _need_source()
    base = _load_position(ns, 120)
    assert ns["identify_base_system"](base) == "ieee9"
    outages = [o for o in ns["enumerate_n1_line_outages"](base) if not o["is_bridge"]]
    assert outages
    for o in outages:
        post, info = ns["solve_n1_case"](base, o["name"], "ieee9")
        assert info["status"] in ("solved", "non_converged", "sanity_rejected"), o
        if info["status"] == "non_converged":
            assert post is None and info["converged"] is False and info["nonconverged_snapshots"], (o, info)
        if info["status"] == "sanity_rejected":
            assert post is None and info["sanity_reason"], (o, info)
        if info["status"] == "solved":
            assert o["name"] not in post.lines.index
            v = post.buses_t.v_mag_pu.values
            assert np.all(np.isfinite(v)) and np.all((v > 0.5) & (v < 1.5))
            remaining = base.lines.index.drop(o["name"])
            assert post.lines.s_nom.loc[remaining].equals(base.lines.s_nom.loc[remaining])
    assert o["name"] in base.lines.index   # base net untouched (copy semantics)


def _line_bridges_independent(net):
    """Line names whose removal disconnects the bus graph (lines + transformers); independent of nx.bridges."""
    import networkx as nx
    out = set()
    for name in net.lines.index:
        G = nx.Graph()
        G.add_nodes_from(net.buses.index)
        G.add_edges_from(zip(net.lines.drop(name).bus0, net.lines.drop(name).bus1))
        G.add_edges_from(zip(net.transformers.bus0, net.transformers.bus1))
        if not nx.is_connected(G):
            out.add(name)
    return out


@pytest.fixture(scope="module")
def e2e(ns, tmp_path_factory):
    _need_source()
    out_dir = tmp_path_factory.mktemp("n1_out")
    summary = ns["generate_n1_contingency_dataset"](SOURCE_INDEX, 1, "pytest", out_dir=str(out_dir))
    header, df = ns["read_n1_sidecar"](summary["meta_path"])
    base = _load_position(ns, 138)
    return dict(out_dir=str(out_dir), summary=summary, header=header, df=df, base=base)


def test_e2e_ieee30_rows_and_files(ns, e2e):
    base, df, summary = e2e["base"], e2e["df"], e2e["summary"]
    assert ns["identify_base_system"](base) == "ieee30"
    assert e2e["header"]["tag"] == "n1_mixed_1500_w_pytest"
    assert len(df) == len(base.lines)
    assert list(df.outage_name) == list(base.lines.index)
    assert set(df.source_position) == {138}
    assert (df.status == "bridge_skipped").sum() == len(_line_bridges_independent(base))
    assert set(df.loc[df.status == "bridge_skipped", "outage_name"]) == _line_bridges_independent(base)
    solved = df[df.status == "solved"]
    k = len(solved)
    assert k > 0
    assert [int(p) for p in solved.file_position] == list(range(k))
    assert df.loc[df.status != "solved", "file_position"].isna().all()
    with open(summary["index_path"], encoding="utf-8") as fh:
        index = json.load(fh)
    assert len(index["files"]) == k
    assert index["tag"] == "n1_mixed_1500_w_pytest"
    pypsa = ns["pypsa"]
    for _, row in solved.iterrows():
        net = pypsa.Network(os.path.join(e2e["out_dir"], index["files"][int(row.file_position)]))
        assert row.outage_name not in net.lines.index
        assert {"s_limit_pu", "limit_source"} <= set(net.lines.columns)
    nonconv = df[df.status == "non_converged"]
    assert all(len(s) > 0 for s in nonconv.nonconverged_snapshots)
    assert sum(sum(v.values()) for v in summary["counts"].values()) == len(df)
    for f in ("base_nonconverged_snapshots", "base_sanity_failed_snapshots"):   # [STSI 151026]: task 092
        assert f in df.columns and all(isinstance(v, list) for v in df[f]), f
        assert all(v == [] for v in df[f]), f   # research 2026-10-15 R1: pos 138 base fully converged + sane
    assert e2e["header"]["base_check"]["pf_resolve"] is False
    assert e2e["header"]["base_check"]["sanity"] == dict(ns["N1_PF_ACCEPTANCE"])


# [STSI 151026]: task 092 - per-snapshot base check (functional on injected faults + source guard)
def test_base_snapshot_check_clean_and_injected(ns):
    _need_source()
    base = _load_position(ns, 120)
    assert ns["identify_base_system"](base) == "ieee9"
    assert ns["base_snapshot_check"](base, "ieee9") == dict(base_nonconverged_snapshots=[],
                                                           base_sanity_failed_snapshots=[])
    bad = base.copy()
    snaps = list(bad.snapshots)
    vm = bad.buses_t.v_mag_pu.copy(); vm.iloc[3, 0] = np.nan; vm.iloc[7, 0] = 1.6; bad.buses_t["v_mag_pu"] = vm
    p0 = bad.lines_t.p0.copy(); p0.iloc[5, 0] = np.nan; bad.lines_t["p0"] = p0
    before = bad.buses_t.v_mag_pu.copy()
    got = ns["base_snapshot_check"](bad, "ieee9")
    assert got["base_nonconverged_snapshots"] == [str(snaps[3]), str(snaps[5])]   # V NaN, line flow NaN
    assert got["base_sanity_failed_snapshots"] == [str(snaps[3]), str(snaps[7])]  # NaN V, V=1.6 > 1.5; NaN p0 passes sanity
    pd.testing.assert_frame_equal(bad.buses_t.v_mag_pu, before)
    assert len(bad.snapshots) == 24


def test_base_snapshot_check_source_guard():
    fn = _func_src(_cell_source(GEN_MARKER), "base_snapshot_check")
    lim = [l.strip() for l in fn.splitlines() if l.strip().startswith("# Limitation")]
    assert len(lim) == 1, lim
    assert "near-converged unfinished NR state is not detected" in lim[0]
    for forbidden in (".pf(", ".lpf(", "solve_power_flow_with_slack("):
        assert forbidden not in fn, forbidden
    assert "copy(snapshots=[s])" in fn
    assert "**N1_PF_ACCEPTANCE" in fn


def test_e2e_header_line_limits_reproduce_attached_limits(ns, e2e):
    header, df, base = e2e["header"], e2e["df"], e2e["base"]
    ll = header["line_limits"]
    real = {s: {n: (v, src) for n, (v, src) in d.items()} for s, d in ll["real"].items()}
    synth = {(s, n): v for s, n, v in ll["synthetic"]}
    with open(e2e["summary"]["index_path"], encoding="utf-8") as fh:
        files = json.load(fh)["files"]
    pypsa = ns["pypsa"]
    for _, row in df[df.status == "solved"].iterrows():
        exp = base.copy()
        exp.remove("Line", row.outage_name)
        ns["attach_line_limits"](exp, row.system, real[row.system], synth)
        got = pypsa.Network(os.path.join(e2e["out_dir"], files[int(row.file_position)]))
        e = exp.lines.loc[got.lines.index]
        assert list(got.lines.index) == list(exp.lines.index)
        np.testing.assert_array_equal(got.lines.s_limit_pu.values.astype(float), e.s_limit_pu.values.astype(float))
        assert list(got.lines.limit_source) == list(e.limit_source)


def _snapshot(paths_root, paths):
    snap = {}
    for p in paths:
        snap[p] = (os.path.getmtime(p), _sha256(p))
    sub = os.path.join(paths_root)
    for dirpath, _, names in os.walk(sub):
        for n in names:
            fp = os.path.join(dirpath, n)
            snap[fp] = (os.path.getmtime(fp), _sha256(fp))
    return snap


def test_e2e_second_call_raises_and_leaves_outputs_untouched(ns, e2e):
    s = e2e["summary"]
    sub_dir = os.path.join(e2e["out_dir"], e2e["header"]["tag"])
    before = _snapshot(sub_dir, [s["index_path"], s["meta_path"]])
    with pytest.raises(FileExistsError):
        ns["generate_n1_contingency_dataset"](SOURCE_INDEX, 1, "pytest", out_dir=e2e["out_dir"])
    assert _snapshot(sub_dir, [s["index_path"], s["meta_path"]]) == before


def test_zz_source_index_unchanged():
    _need_source()
    assert _sha256(SOURCE_INDEX) == SOURCE_SHA_BEFORE
