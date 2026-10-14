"""Task 088: verbatim-copy parity for every `# [COPY of <notebook>::<name>]` marker in the Contingency and DataGen
notebooks. Origin notebooks are only read (json.load); the copied block must equal the origin's last top-level
definition byte for byte (first decorator line .. end_lineno).
"""
# [STSI 141026]:new copy-parity test for the contingency notebook (task 088)
import ast
import functools
import json
import os
import re

REPO = os.path.dirname(os.path.abspath(__file__))
CONTINGENCY = os.path.join(REPO, "GNN_Powerflow_V2.7_Contingency.ipynb")
DATAGEN = os.path.join(REPO, "GNN_Powerflow_V2.6_DataGen.ipynb")
ORIGINS = ("GNN_Powerflow_V2.6_DataGen", "GNN_Powerflow_V2.7_Training", "GNN_Powerflow_V2.7_Analysis")

MARKER = re.compile(r"^# \[COPY of (GNN_Powerflow_V[\w.]+)::(\w+)\]\s*$")
EXC = re.compile(r"^# PRECEDENCE EXCEPTION \(D13( scope extension, user 2026-10-14)?\): "
                 r"by-rule origin GNN_Powerflow_V2\.7_Training$")
D13_NOTE = "# NOTE: deliberate exception to DataGen>Training>Analysis precedence (user decision D13, 2026-10-14)."

_DG, _TR, _AN = ORIGINS
EXPECTED = (
    {(_DG, n) for n in ("load_dataset_list", "load_dataset_subset", "read_n1_sidecar", "attach_line_limits")}
    | {(_TR, n) for n in ("PowerFlowGNN", "PhysicsConfig", "compute_ptdf_matrix")}
    | {(_AN, n) for n in ("precompute_bfs_order", "reconstruct_theta_from_delta",
                          "reconstruct_theta_from_path_matrix", "PowerFlowDataset", "collate_with_ptdf",
                          "calculate_line_flows", "calculate_line_flows_from_delta_theta",
                          "_augment_edge_attr_learned", "predict_network_results_with_masks",
                          "_compute_delta_theta_metrics", "postprocess_dangling_buses", "evaluate_gnn_on_test_set",
                          "load_sweep_results", "pypsa_to_ppc", "evaluate_pypower_baseline",
                          "evaluate_fdlf_baseline", "evaluate_pandapower_baseline", "build_ref_baselines",
                          "evaluate_pypsa_timing", "evaluate_pypsa_timing_single", "evaluate_dc_baseline",
                          "predict_network_results_dc", "reconstruct_test_split")}
)
EXPECTED_EXC = {"PowerFlowDataset": None, "collate_with_ptdf": None,
                "precompute_bfs_order": " scope extension, user 2026-10-14",
                "reconstruct_theta_from_delta": " scope extension, user 2026-10-14",
                "calculate_line_flows": " scope extension, user 2026-10-14",
                "calculate_line_flows_from_delta_theta": " scope extension, user 2026-10-14"}


def _clean(src):
    """Comment out IPython magic / shell lines; line numbers are preserved."""
    out = []
    for line in src.split("\n"):
        st = line.lstrip()
        out.append(line[: len(line) - len(st)] + "#" + st if st.startswith(("%", "!")) else line)
    return "\n".join(out)


def _block(lines, node):
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    return "\n".join(lines[start - 1:node.end_lineno])


def _code_sources(nb_path):
    with open(nb_path, encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in cells if c["cell_type"] == "code"]


_DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@functools.lru_cache(maxsize=None)
def _top_defs(nb_path):
    """({name: block_text} with the last top-level definition winning, {name in an unparsable cell})."""
    defs, unparsable = {}, set()
    for src in _code_sources(nb_path):
        try:
            tree = ast.parse(_clean(src))
        except SyntaxError:
            unparsable |= set(re.findall(r"^(?:async\s+def|def|class)\s+(\w+)", src, re.M))
            continue
        lines = src.split("\n")
        for n in tree.body:
            if isinstance(n, _DEF):
                defs[n.name] = _block(lines, n)
    return defs, frozenset(unparsable)


def _origin_path(origin):
    return os.path.join(REPO, origin + ".ipynb")


@functools.lru_cache(maxsize=None)
def _markers(nb_path):
    """[(origin, name, marker_line_idx, cell_lines, top_nodes)] for every marker in the notebook's code cells."""
    out = []
    for src in _code_sources(nb_path):
        lines = src.split("\n")
        hits = [(i, MARKER.match(l)) for i, l in enumerate(lines)]
        hits = [(i, m) for i, m in hits if m]
        if not hits:
            continue
        nodes = ast.parse(_clean(src)).body
        for i, m in hits:
            out.append((m.group(1), m.group(2), i, lines, nodes))
    return out


def _check_markers(nb_path):
    found = _markers(nb_path)
    assert found, f"no markers in {nb_path}"
    for origin, name, i, lines, nodes in found:
        nxt = [n for n in nodes if isinstance(n, _DEF)
               and min([d.lineno for d in n.decorator_list] + [n.lineno]) == i + 2]  # 1-based line i+2 == idx i+1
        assert len(nxt) == 1, f"{origin}::{name}: no def/class starts directly below its marker"
        assert nxt[0].name == name, f"marker {name!r} is followed by {nxt[0].name!r}"
        defs, unparsable = _top_defs(_origin_path(origin))
        assert name not in unparsable, f"{origin}::{name} also defined in an unparsable origin cell"
        assert name in defs, f"{origin}::{name} not defined in origin"
        assert _block(lines, nxt[0]) == defs[name], f"{origin}::{name} drifted from its origin"
    return {(o, n) for o, n, *_ in found}


def test_contingency_copies_verbatim():
    assert len(_check_markers(CONTINGENCY)) == 30


def test_contingency_expected_copy_set():
    found = [(o, n) for o, n, *_ in _markers(CONTINGENCY)]
    assert len(found) == len(set(found)) == 30
    assert set(found) == EXPECTED


def test_precedence_exceptions_marked():
    got = {}
    for _, name, i, lines, _ in _markers(CONTINGENCY):
        m = EXC.match(lines[i - 1]) if i > 0 else None
        if m:
            got[name] = m.group(1)
    assert got == EXPECTED_EXC
    # no stray exception lines anywhere else
    n_exc = sum(1 for src in _code_sources(CONTINGENCY) for l in src.split("\n") if l.startswith("# PRECEDENCE"))
    assert n_exc == 6


def test_datagen_copies_verbatim():
    assert _check_markers(DATAGEN) == {("GNN_Powerflow_V2.7_Analysis", "reconstruct_test_split")}


def test_no_unmarked_shadowing():
    origin_names = set()
    for o in ORIGINS:
        defs, unparsable = _top_defs(_origin_path(o))
        origin_names |= set(defs) | set(unparsable)
    marked = {n for _, n, *_ in _markers(CONTINGENCY)}
    top = set()
    for src in _code_sources(CONTINGENCY):
        top |= {n.name for n in ast.parse(_clean(src)).body if isinstance(n, _DEF)}
    unmarked = (top & origin_names) - marked
    assert not unmarked, f"unmarked copies/shadows of origin names: {sorted(unmarked)}"


def test_d13_note_present():
    srcs = _code_sources(CONTINGENCY)
    l3 = [s for s in srcs if "# [COPY of GNN_Powerflow_V2.7_Analysis::PowerFlowDataset]" in s]
    assert len(l3) == 1
    assert l3[0].split("\n")[0] == D13_NOTE
    assert sum(s.count(D13_NOTE) for s in srcs) == 1
