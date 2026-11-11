# [STSI 081126]:cover signed-angle diagnostics with in-memory networks only; tests never invoke PyPSA power flow.
import json
import sys

import numpy as np
import pandas as pd
import pytest

from gnn_powerflow import dtheta_diagnostics as diag


class Table:
    def __init__(self, **frames):
        self.__dict__.update(frames)


class FakeNetwork:
    def __init__(self, angles, lines=None, transformers=None, v=None, p=None, q=None, loads=None,
                 p_set=None, q_set=None):
        self.lines = lines if lines is not None else pd.DataFrame(columns=["bus0", "bus1"])
        self.transformers = transformers if transformers is not None else pd.DataFrame(columns=["bus0", "bus1"])
        self.buses = pd.DataFrame(index=angles.columns)
        self.buses_t = Table(v_ang=angles, v_mag_pu=v if v is not None else pd.DataFrame(1.0, index=angles.index, columns=angles.columns),
                             p=p if p is not None else pd.DataFrame(index=angles.index),
                             q=q if q is not None else pd.DataFrame(index=angles.index))
        self.loads = loads if loads is not None else pd.DataFrame(columns=["bus"])
        self.loads_t = Table(p_set=p_set if p_set is not None else pd.DataFrame(index=angles.index),
                             q_set=q_set if q_set is not None else pd.DataFrame(index=angles.index))


def _network(values, reverse=False, extra_transformer=True):
    a = pd.DataFrame({"a": np.radians(values), "b": np.zeros(len(values)), "c": np.zeros(len(values))})
    lines = pd.DataFrame({"bus0": ["a", "a"], "bus1": ["b", "b"]})
    transformers = pd.DataFrame({"bus0": ["a"], "bus1": ["b"]}) if extra_transformer else pd.DataFrame(columns=["bus0", "bus1"])
    if reverse:
        lines[["bus0", "bus1"]] = lines[["bus1", "bus0"]].to_numpy()
        if extra_transformer:
            transformers[["bus0", "bus1"]] = transformers[["bus1", "bus0"]].to_numpy()
    return FakeNetwork(a, lines, transformers)


def test_orientation_lines_transformers_parallel_and_sign_reversal():
    forward = diag.analyze_network(_network([10, -5]), "ieee9")
    reverse = diag.analyze_network(_network([10, -5], reverse=True), "ieee9")
    assert forward["branch_count"] == 3
    assert forward["line_count"] == 2 and forward["transformer_count"] == 1
    assert forward["finite_signed_count"] == 6
    assert np.array_equal(forward["delta"], -reverse["delta"])
    assert forward["bin_counts"].sum() == 6


def test_under_overflow_exact_edges_and_zero_are_accounted_once():
    net = _network([-85, -80, -75, -5, 0, 5, 75, 80, 90])
    out = diag.analyze_network(net)
    assert out["finite_signed_count"] == 27
    assert out["zero_count"] == 3
    assert out["bin_counts"].sum() == 27
    assert out["bin_counts"][0] == 3
    assert out["bin_counts"][-1] == 6
    assert out["bin_counts"][1] == 3
    assert out["bin_counts"][2] == 3


def test_nonfinite_and_missing_bus_lookup_are_explicit():
    net = _network([1, np.nan, np.inf])
    out = diag.analyze_network(net)
    assert out["finite_signed_count"] == 3
    assert out["nonfinite_signed_count"] == 6
    missing = _network([1, 2])
    missing.lines.loc[0, "bus1"] = "not-a-bus"
    bad = diag.analyze_network(missing)
    assert bad["snapshot_count"] == 2 and bad["branch_count"] == 3
    assert bad["finite_signed_count"] == 0
    assert bad["nonfinite_signed_count"] == 6
    assert np.array_equal(bad["bin_counts"], np.zeros(len(diag.BIN_LABELS), dtype=int))
    assert np.array_equal(bad["bin_fractions"], np.zeros(len(diag.BIN_LABELS)))
    assert bad["delta"].size == 0
    assert all(values.size == 0 for values in bad["branch_type_values"].values())
    assert bad["zero_count"] == bad["negative_count"] == bad["positive_count"] == 0
    assert bad["max_abs_delta_deg"] is None
    assert bad["available"]["angles"] is False
    assert len(bad["errors"]) == 2
    assert bad["errors"][0] == "branch bus lookup missing from buses_t.v_ang columns: ['not-a-bus']"
    assert bad["errors"][1].startswith("bus lookup / angle conversion failed: KeyError:")


@pytest.mark.parametrize("missing_bus", [False, True])
def test_bus_lookup_controls_system_completeness(tmp_path, monkeypatch, missing_bus):
    net = _network([1, 2])
    net.buses_t.p = net.buses_t.q = net.buses_t.v_mag_pu.copy()
    if missing_bus:
        net.lines.loc[0, "bus1"] = "not-a-bus"
    monkeypatch.setitem(sys.modules, "pypsa", Table(Network=lambda path: net))
    index = tmp_path / "synthetic.json"
    index.write_text(json.dumps({"files": ["network.nc"]}), encoding="utf-8")
    (tmp_path / "network.nc").write_bytes(b"synthetic; not a NetCDF input")
    (tmp_path / "synthetic_meta.json").write_text(json.dumps({
        "n_attempts": 1, "n_accepted": 1,
        "rows": [{"status": "accepted_full", "file_position": 0, "n_snapshots": 2, "n_ok": 2}],
    }), encoding="utf-8")
    result, rows = diag._analyze_system("synthetic", index, {}, {})
    assert result["sidecar"]["reconciled"]
    assert result["status"] == ("incomplete" if missing_bus else "complete")
    assert rows[0]["finite_signed_count"] == (0 if missing_bus else 6)
    assert rows[0]["nonfinite_signed_count"] == (6 if missing_bus else 0)
    assert rows[0]["bin_counts"].sum() == (0 if missing_bus else 6)
    assert result["summary"]["finite_branch_snapshots"] == (0 if missing_bus else 6)
    if missing_bus:
        assert len(result["errors"]) == 3
        assert result["errors"][-1] == "network[0] network.nc: 6 nonfinite/missing signed values of 6"
    else:
        assert result["errors"] == []


def test_empty_voltage_snapshots_match_plot_exclusion():
    angle = pd.DataFrame({"a": [0.2], "b": [0.0]})
    net = FakeNetwork(angle, lines=pd.DataFrame({"bus0": ["a"], "bus1": ["b"]}),
                      v=pd.DataFrame(index=angle.index))
    out = diag.analyze_network(net)
    assert out["finite_signed_count"] == 0
    assert out["snapshot_count"] == 0
    assert any("excluded from plotted baseline" in error for error in out["errors"])


def test_network_weighting_differs_from_pooled_weighting_and_top_cohort_mixes():
    smalls = [diag.analyze_network(_network([0]), "x", index_position=i) for i in range(3)]
    large = diag.analyze_network(_network([0, 20]), "x", index_position=3)
    summary = diag._summarize_system("x", [*smalls, large], [*smalls, large])
    zero_bin = diag.BIN_LABELS.index("[0,5)")
    twenty_bin = diag.BIN_LABELS.index("[20,25)")
    assert smalls[0]["bin_fractions"][zero_bin] == 1
    assert large["bin_fractions"][twenty_bin] == 0.5
    assert summary["bin_counts"][zero_bin] == 12
    assert summary["bin_counts"][twenty_bin] == 3
    assert summary["cohort_sizes"] == {"0": 3, "1": 0, "2": 0, "3": 1}
    assert summary["top_cohort"]["fraction_in_minus5_to_plus5"] == 0.5


def test_load_setpoint_fallback_is_identified_for_p_and_q():
    angle = pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]})
    loads = pd.DataFrame({"bus": ["a", "a"]}, index=["la", "lb"])
    net = FakeNetwork(angle, loads=loads, p_set=pd.DataFrame({"la": [3.0], "lb": [4.0]}),
                      q_set=pd.DataFrame({"la": [5.0], "lb": [6.0]}))
    result = diag.analyze_network(net)
    assert "fallback" in result["margins"]["P"]["source"]
    assert result["margins"]["P"]["count"] == 2
    assert np.array_equal(result["margin_values"]["P"], [3.0, 4.0])
    assert result["margins"]["Q"]["count"] == 2


def test_sidecar_position_reconciliation_and_rejected_bins_unavailable():
    sidecar = {"n_attempts": 2, "n_accepted": 1, "seed": 42, "rows": [
        {"status": "accepted_full", "file_position": 0, "n_snapshots": 24, "n_ok": 24,
         "draws": {"lam_bucket": 1}},
        {"status": "rejected_structural", "reason": "structural", "n_snapshots": 24, "n_ok": 0, "failed": {},
         "draws": {"lam_bucket": 1}},
    ], "snapshot_reason_counts": {"ok": 24}}
    summary = diag._sidecar_summary(sidecar, ["n0.nc"], "ieee9")
    assert summary["reconciled"]
    assert summary["accepted_snapshots"] == 24
    assert summary["declared_n_snapshots"] == 48
    assert summary["evaluated_snapshots"] == 24
    assert summary["lambda_bucket_rows"]["1"]["attempts"] == 2
    assert "unavailable" in summary["rejected_signed_bin_attribution"]
    bad = diag._sidecar_summary(sidecar, ["n0.nc", "n1.nc"], "ieee9")
    assert not bad["reconciled"]


def test_missingness_and_six_system_gap_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "NOTEBOOK_PATH", tmp_path / "notebook.ipynb")
    (tmp_path / "notebook.ipynb").write_text("test", encoding="utf-8")
    outdir = tmp_path / "report"
    result = diag._write_outputs(outdir, tmp_path, {}, False)
    assert set(result["systems"]) == set(diag.SYSTEMS)
    assert all(result["systems"][s]["status"] == "incomplete" for s in diag.SYSTEMS)
    assert all((outdir / name).is_file() for name in diag.OUTPUT_FILES)
    assert len(list(csv_rows(outdir / "signed_bins.csv"))) == len(diag.SYSTEMS) * len(diag.BIN_LABELS)


def csv_rows(path):
    import csv
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_no_overwrite_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diag._write_outputs(output, tmp_path, {}, False)
    assert marker.read_text(encoding="utf-8") == "unchanged"


# [STSI 091126]:verify pooled target bins, uncapped tails and solved all-bus/PQ voltage edges.
def test_coordinated_comparison_bins_keep_full_denominators_and_edges():
    angle = pd.DataFrame({
        "a": np.radians([0.0, 0.9, 1.0, 9.0, 10.0, 10.01, 31.0]),
        "b": np.zeros(7),
    })
    net = FakeNetwork(
        angle,
        lines=pd.DataFrame({"bus0": ["a"], "bus1": ["b"]}),
        transformers=pd.DataFrame(columns=["bus0", "bus1"]),
        v=pd.DataFrame({"a": [0.90, 0.949, 0.95, 1.0, 1.05, 1.10, 1.101],
                        "b": [0.90, 0.949, 0.95, 1.0, 1.05, 1.10, 1.101]}),
    )
    net.buses["control"] = ["PQ", "PV"]

    metrics = diag.comparison_network_metrics(net, (0.90, 1.10))
    angle_hist = diag._target_angle_histogram(metrics["angle_values"])
    assert angle_hist["counts"] == [2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2]
    assert angle_hist["finite_count"] == 7
    assert angle_hist["in_target_count"] == 5
    assert angle_hist["over_10_count"] == 2
    assert angle_hist["conditional_0_10_fractions"][-1] == pytest.approx(2 / 5)

    voltage = diag._voltage_histogram(metrics["all_bus_v_values"], (0.90, 1.10))
    assert voltage["inside_window_count"] == 12
    assert voltage["outside_window_count"] == 2
    assert voltage["counts"][-1] == 2
    assert voltage["edges"][0] == 0.90 and voltage["edges"][-1] == 1.10
    assert voltage["counts"][1] == 2
    pq = diag._voltage_histogram(metrics["pq_bus_v_values"], (0.90, 1.10))
    assert pq["finite_count"] == 7
    assert pq["inside_window_count"] == 6


def test_comparison_aggregation_preserves_parallel_and_branch_type_denominators():
    net = _network([0.0, 10.0, 10.01], extra_transformer=True)
    summary = diag.aggregate_comparison_metrics([net], (0.8, 1.2))
    pooled = summary["branch_snapshot_observations"]
    assert summary["interpretation"] == "PF-accepted, not equipment-feasible"
    assert summary["line_count_per_network_total"] == 2
    assert summary["transformer_count_per_network_total"] == 1
    assert pooled["finite_count"] == 9
    assert pooled["in_target_count"] == 6
    assert pooled["over_10_count"] == 3
    assert sum(pooled["unconditional_fractions"]) == pytest.approx(1.0)
    assert summary["line_branch_snapshot_observations"]["finite_count"] == 6
    assert summary["transformer_branch_snapshot_observations"]["finite_count"] == 3
    assert 0.0 <= summary["D_theta"] <= 1.0 and 0.0 <= summary["D_V"] <= 1.0
    assert summary["D"] == pytest.approx(0.5 * summary["D_theta"] + 0.5 * summary["D_V"])
