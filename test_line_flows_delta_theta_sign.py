"""Task 096 (F7): signed p0/q0/p1/q1 of calculate_line_flows_from_delta_theta vs PyPSA (source-only:
notebooks are only json-read; just the function source is exec'd)."""
# [STSI 191026]:task 096 - end-1 sign regression test
import ast, json, os
import numpy as np
import pypsa
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
NAME = "calculate_line_flows_from_delta_theta"
NOTEBOOKS = ["GNN_Powerflow_V2.7_Analysis.ipynb", "GNN_Powerflow_V2.7_Training.ipynb",
             "GNN_Powerflow_V2.7_Contingency.ipynb"]


def _load(nb_file):
    """Last top-level definition of NAME (same rule as test_contingency_copy_parity.py)."""
    with open(os.path.join(REPO, nb_file), encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    block = None
    for c in cells:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c["cell_type"] != "code" or f"def {NAME}(" not in src:
            continue
        lines = src.split("\n")
        clean = "\n".join("#" + l if l.lstrip().startswith(("%", "!")) else l for l in lines)
        for n in ast.parse(clean).body:
            if isinstance(n, ast.FunctionDef) and n.name == NAME:
                block = "\n".join(lines[n.lineno - 1:n.end_lineno])
    assert block is not None, nb_file
    g = {"np": np}
    exec(compile(block, f"<{nb_file}:{NAME}>", "exec"), g)  # noqa: S102
    return g[NAME]


@pytest.fixture(scope="module")
def net():
    n = pypsa.Network()
    n.set_snapshots(range(2))
    for b in ("b0", "b1", "b2"):
        n.add("Bus", b, v_nom=1.0)   # flat pu: impedances and powers are already pu
    n.add("Generator", "g0", bus="b0", control="Slack")
    n.add("Generator", "g1", bus="b1", control="PV", p_set=0.3)
    n.add("Load", "d2", bus="b2", p_set=[0.5, 0.8], q_set=[0.2, 0.3])
    for name, b0, b1, r, x, b in (("L01", "b0", "b1", 0.01, 0.10, 0.04), ("L12", "b1", "b2", 0.02, 0.15, 0.03),
                                  ("L02", "b0", "b2", 0.015, 0.12, 0.05)):
        n.add("Line", name, bus0=b0, bus1=b1, r=r, x=x, b=b)
    assert n.pf().converged.all().all()
    return n


@pytest.mark.parametrize("nb_file", NOTEBOOKS)
def test_signed_flows_match_pypsa(nb_file, net):
    f = _load(nb_file)
    idx = {b: i for i, b in enumerate(net.buses.index)}
    ei = np.array([[idx[b] for b in net.lines.bus0], [idx[b] for b in net.lines.bus1]])
    ea = np.column_stack([net.lines.r, net.lines.x, net.lines.b / 2, np.zeros(len(net.lines)),
                          np.zeros(len(net.lines)), np.zeros(len(net.lines))])   # tap 0 -> 1.0 inside
    for s in net.snapshots:
        vm = net.buses_t.v_mag_pu.loc[s, net.buses.index].to_numpy(float)
        va = net.buses_t.v_ang.loc[s, net.buses.index].to_numpy(float)
        fl = f(va[ei[0]] - va[ei[1]], vm, ei, ea)
        for k in ("p0", "q0", "p1", "q1"):
            np.testing.assert_allclose(fl[k], getattr(net.lines_t, k).loc[s, net.lines.index].to_numpy(float),
                                       atol=1e-8, err_msg=f"{nb_file} {k} snapshot {s}")
