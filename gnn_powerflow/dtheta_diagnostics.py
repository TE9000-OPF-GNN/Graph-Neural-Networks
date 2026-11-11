# [STSI 081126]:read-only saved-network signed branch-angle diagnostics; no generation or power-flow calls.
"""Read-only diagnostics for saved PyPSA datasets, with reproducible provenance outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SYSTEMS = ("ieee9", "cigre14", "ieee30", "ieee39", "ieee57", "ieee118")
INDEX_NAMES = {system: f"{system}_1500_u1.json" for system in SYSTEMS}
DEFAULT_DATA_ROOT = Path(r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN")
NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "GNN_Powerflow_V2.6_DataGen.ipynb"
BIN_EDGES = np.arange(-80.0, 85.0, 5.0)
BIN_LABELS = ["< -80"] + [f"[{lo:g},{hi:g})" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])] + [">= 80"]
OUTPUT_FILES = ("report.md", "summary.json", "signed_bins.csv", "network_stats.csv")


def _sha256(path: Path) -> str | None:
    """Hash a source file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "sha256": None, "error": "path not identified"}
    try:
        return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size, "error": None}
    except Exception as exc:
        return {"path": str(path.resolve(strict=False)), "sha256": None, "size_bytes": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _array(frame: Any) -> np.ndarray:
    if frame is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(frame, pd.DataFrame):
        return frame.to_numpy(dtype=float, copy=True)
    return np.asarray(frame, dtype=float)


def _series_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return {"count": 0, "missing_or_nonfinite": int(arr.size), "min": None, "max": None,
                "quantiles": {str(q): None for q in (1, 5, 25, 50, 75, 95, 99)}}
    qs = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99]).tolist()
    return {"count": int(finite.size), "missing_or_nonfinite": int(arr.size - finite.size),
            "min": float(finite.min()), "max": float(finite.max()),
            "quantiles": {str(q): float(v) for q, v in zip((1, 5, 25, 50, 75, 95, 99), qs)}}


def _bin_indices(values: Any) -> np.ndarray:
    """Bin every finite signed angle once; exact edges use half-open interval semantics."""
    x = np.asarray(values, dtype=float)
    index = np.empty(x.shape, dtype=np.int16)
    index[x < -80.0] = 0
    index[x >= 80.0] = len(BIN_LABELS) - 1
    middle = (x >= -80.0) & (x < 80.0)
    index[middle] = np.searchsorted(BIN_EDGES[1:-1], x[middle], side="right") + 1
    return index


def _get_frame(obj: Any, attr: str) -> pd.DataFrame:
    value = getattr(obj, attr, None)
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value
    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _branch_observations(net: Any) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply the plot's literal bus0-minus-bus1 definition, preserving branch row orientation."""
    errors: list[str] = []
    branches = []
    for kind in ("lines", "transformers"):
        frame = _get_frame(net, kind)
        if {"bus0", "bus1"}.issubset(frame.columns):
            part = frame.loc[:, ["bus0", "bus1"]].copy()
            part["type"] = kind[:-1] if kind == "lines" else "transformer"
            branches.append(part.reset_index(drop=True))
    branch_frame = pd.concat(branches, ignore_index=True) if branches else pd.DataFrame(columns=["bus0", "bus1", "type"])
    angle = _get_frame(getattr(net, "buses_t", None), "v_ang")
    if not len(branch_frame) or not len(angle):
        if len(branch_frame) and not len(angle):
            errors.append("buses_t.v_ang missing or empty")
        return np.full((len(angle), len(branch_frame)), np.nan), branch_frame, angle, errors
    # [STSI 081126]:identify every absent branch endpoint before attempting the literal plot lookup.
    missing = sorted((set(branch_frame.bus0) | set(branch_frame.bus1)) - set(angle.columns), key=str)
    if missing:
        errors.append(f"branch bus lookup missing from buses_t.v_ang columns: {missing!r}")
    try:
        left = angle.loc[:, branch_frame.bus0.tolist()].to_numpy(dtype=float)
        right = angle.loc[:, branch_frame.bus1.tolist()].to_numpy(dtype=float)
        delta = np.degrees(left - right)
    except Exception as exc:
        errors.append(f"bus lookup / angle conversion failed: {type(exc).__name__}: {exc}")
        # Preserve the expected denominator without fabricating observations after a failed lookup.
        delta = np.full((len(angle), len(branch_frame)), np.nan)
    return delta, branch_frame, angle, errors


def _margin_frames(net: Any) -> dict[str, tuple[np.ndarray, str]]:
    """Collect bus V/P/Q frames; when bus P/Q is empty, map load setpoints to their buses."""
    bus_t = getattr(net, "buses_t", None)
    out: dict[str, tuple[np.ndarray, str]] = {}
    v = _get_frame(bus_t, "v_mag_pu")
    out["V"] = (_array(v), "buses_t.v_mag_pu")
    for label in ("P", "Q"):
        frame = _get_frame(bus_t, label.lower())
        source = f"buses_t.{label.lower()}"
        if frame.empty:
            loads = _get_frame(getattr(net, "loads_t", None), f"{label.lower()}_set")
            if not loads.empty:
                # [STSI 081126]:match the saved plot's fallback exactly by flattening load setpoints unchanged.
                frame = loads
                source = f"loads_t.{label.lower()}_set flattened (plot fallback)"
            else:
                frame = pd.DataFrame()
                source = f"unavailable: buses_t.{label.lower()} empty and load-setpoint fallback unavailable"
        out[label] = (_array(frame), source)
    return out


# [STSI 091126]:add pooled kept-snapshot angle and solved-voltage metrics for the coordinated comparison.
ABS_DTHETA_EDGES = np.arange(0.0, 11.0, 1.0)


def _target_angle_histogram(values: Any) -> dict[str, Any]:
    """Count all finite absolute branch observations in 1-degree target bins and an uncapped tail."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    inside = finite[(finite >= 0.0) & (finite <= 10.0)]
    tail = finite[finite > 10.0]
    counts = np.zeros(11, dtype=np.int64)
    if inside.size:
        # 10 degrees is deliberately assigned to the closed final target bin.
        index = np.minimum(np.floor(inside).astype(np.int64), 9)
        counts[:10] = np.bincount(index, minlength=10)
    counts[10] = tail.size
    denominator = int(finite.size)
    fractions = counts / denominator if denominator else np.zeros(11, dtype=float)
    conditional = counts[:10] / inside.size if inside.size else np.zeros(10, dtype=float)
    return {
        "labels": [f"[{i},{i + 1})" for i in range(9)] + ["[9,10]"] + [">10"],
        "counts": counts.tolist(),
        "unconditional_fractions": fractions.tolist(),
        "conditional_0_10_fractions": conditional.tolist(),
        "finite_count": denominator,
        "missing_or_nonfinite": int(raw.size - denominator),
        "in_target_count": int(inside.size),
        "in_target_fraction": float(inside.size / denominator) if denominator else None,
        "over_10_count": int(tail.size),
        "over_10_fraction": float(tail.size / denominator) if denominator else None,
        "max": float(finite.max()) if denominator else None,
        "quantiles": ({str(q): float(v) for q, v in zip((1, 5, 25, 50, 75, 95, 99),
                                                            np.percentile(finite, [1, 5, 25, 50, 75, 95, 99]))}
                      if denominator else {str(q): None for q in (1, 5, 25, 50, 75, 95, 99)}),
    }


def _voltage_histogram(values: Any, window: tuple[float, float], bin_width: float = 0.05) -> dict[str, Any]:
    """Histogram solved V values with clipped 0.05-p.u. bins and an explicit outside-window category."""
    if (len(window) != 2 or not all(math.isfinite(float(x)) for x in window)
            or not float(window[0]) < float(window[1]) or bin_width <= 0):
        raise ValueError(f"invalid voltage window/bin width: {window!r}, {bin_width!r}")
    lo, hi = map(float, window)
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    edges = [lo]
    while edges[-1] + bin_width < hi - 1e-12:
        edges.append(round(edges[-1] + bin_width, 12))
    if hi - edges[-1] > 1e-12:
        edges.append(hi)
    else:
        edges[-1] = hi
    edges = np.asarray(edges, dtype=float)
    inside_mask = (finite >= lo) & (finite <= hi)
    inside = finite[inside_mask]
    # right-side search makes interior edges left-closed/right-open; cap hi into the final bin.
    indices = np.searchsorted(edges, inside, side="right") - 1
    indices = np.minimum(indices, len(edges) - 2)
    counts = np.bincount(indices, minlength=len(edges) - 1)
    outside = int(finite.size - inside.size)
    full_counts = np.concatenate((counts, np.asarray([outside], dtype=np.int64)))
    denominator = int(finite.size)
    return {
        "edges": edges.tolist(),
        "labels": [f"[{a:.6g},{b:.6g}{']' if i == len(edges) - 2 else ')'}"
                   for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))] + ["outside_window"],
        "counts": full_counts.tolist(),
        "unconditional_fractions": (full_counts / denominator).tolist() if denominator else
                                   np.zeros(len(full_counts), dtype=float).tolist(),
        "finite_count": denominator,
        "missing_or_nonfinite": int(raw.size - denominator),
        "inside_window_count": int(inside.size),
        "outside_window_count": outside,
        "outside_window_fraction": float(outside / denominator) if denominator else None,
        "outside_window_values": _series_stats(finite[~inside_mask]),
        "min": float(finite.min()) if denominator else None,
        "max": float(finite.max()) if denominator else None,
        "quantiles": ({str(q): float(v) for q, v in zip((1, 5, 25, 50, 75, 95, 99),
                                                            np.percentile(finite, [1, 5, 25, 50, 75, 95, 99]))}
                      if denominator else {str(q): None for q in (1, 5, 25, 50, 75, 95, 99)}),
    }


def comparison_network_metrics(net: Any, voltage_window: tuple[float, float]) -> dict[str, Any]:
    """Summarize actual kept-snapshot absolute branch angles and solved all-bus/PQ voltage values."""
    signed, branches, angle, errors = _branch_observations(net)
    magnitude = np.abs(signed)
    finite_mask = np.isfinite(magnitude)
    line_mask = branches["type"].to_numpy() == "line" if len(branches) else np.array([], dtype=bool)
    transformer_mask = branches["type"].to_numpy() == "transformer" if len(branches) else np.array([], dtype=bool)
    v_frame = _get_frame(getattr(net, "buses_t", None), "v_mag_pu")
    all_v = _array(v_frame).reshape(-1)
    controls = _get_frame(net, "buses")
    pq_columns = [column for column in v_frame.columns
                  if column in controls.index and str(controls.at[column, "control"]).upper() == "PQ"] \
        if "control" in controls.columns else []
    pq_v = _array(v_frame.loc[:, pq_columns]).reshape(-1) if pq_columns else np.array([], dtype=float)
    if "control" not in controls.columns:
        errors.append("buses.control unavailable; PQ-bus Vmag could not be classified")
    else:
        unclassified = [column for column in v_frame.columns
                        if column not in controls.index or pd.isna(controls.at[column, "control"])
                        or not str(controls.at[column, "control"]).strip()]
        if unclassified:
            errors.append(f"bus control unavailable for voltage columns: {unclassified!r}")
    # [STSI 091126]:retain source row identity and bus0/bus1 orientation for parallel-branch diagnostics.
    branch_definitions = []
    for kind in ("lines", "transformers"):
        frame = _get_frame(net, kind)
        if {"bus0", "bus1"}.issubset(frame.columns):
            branch_definitions.extend(
                {"type": "line" if kind == "lines" else "transformer", "branch": str(name),
                 "bus0": str(row["bus0"]), "bus1": str(row["bus1"])}
                for name, row in frame.iterrows())
    return {
        "angle_values": magnitude[finite_mask].reshape(-1),
        "angle_missing_or_nonfinite": int(magnitude.size - int(finite_mask.sum())),
        "line_angle_values": magnitude[:, line_mask][np.isfinite(magnitude[:, line_mask])].reshape(-1)
        if line_mask.any() else np.array([], dtype=float),
        "transformer_angle_values": magnitude[:, transformer_mask][np.isfinite(magnitude[:, transformer_mask])].reshape(-1)
        if transformer_mask.any() else np.array([], dtype=float),
        "all_bus_v_values": all_v,
        "pq_bus_v_values": pq_v,
        "bus_snapshot_observation_count": int(all_v.size),
        "pq_bus_snapshot_observation_count": int(pq_v.size),
        "branch_count": int(len(branches)),
        "line_count": int(line_mask.sum()),
        "transformer_count": int(transformer_mask.sum()),
        "snapshot_count": int(len(angle)),
        "branch_definitions": branch_definitions,
        "errors": errors,
    }


def aggregate_comparison_metrics(networks: Any, voltage_window: tuple[float, float]) -> dict[str, Any]:
    """Pool branch-snapshot and solved-bus observations while retaining per-network secondary shares."""
    angle_values: list[np.ndarray] = []
    line_values: list[np.ndarray] = []
    transformer_values: list[np.ndarray] = []
    all_v_values: list[np.ndarray] = []
    pq_v_values: list[np.ndarray] = []
    network_weighted_target: list[np.ndarray] = []
    branch_definitions_by_network: list[dict[str, Any]] = []
    missing_angles = total_snapshots = branch_count = line_count = transformer_count = 0
    errors: list[str] = []
    for position, network in enumerate(networks):
        metric = comparison_network_metrics(network, voltage_window)
        av = metric["angle_values"]
        angle_values.append(av)
        line_values.append(metric["line_angle_values"])
        transformer_values.append(metric["transformer_angle_values"])
        all_v_values.append(metric["all_bus_v_values"])
        pq_v_values.append(metric["pq_bus_v_values"])
        one = _target_angle_histogram(av)
        network_weighted_target.append(np.asarray(one["unconditional_fractions"], dtype=float))
        missing_angles += metric["angle_missing_or_nonfinite"]
        total_snapshots += metric["snapshot_count"]
        branch_count += metric["branch_count"]
        line_count += metric["line_count"]
        transformer_count += metric["transformer_count"]
        branch_definitions_by_network.append({"network_position": position,
                                              "branches": metric["branch_definitions"]})
        errors.extend(f"network[{position}]: {error}" for error in metric["errors"])
    pooled_angles = np.concatenate(angle_values) if angle_values else np.array([], dtype=float)
    pooled_v = np.concatenate(all_v_values) if all_v_values else np.array([], dtype=float)
    pooled_pq = np.concatenate(pq_v_values) if pq_v_values else np.array([], dtype=float)
    pooled = _target_angle_histogram(pooled_angles)
    pooled_denominator = pooled["finite_count"]
    net_share_mean = (np.mean(np.stack(network_weighted_target), axis=0).tolist()
                      if network_weighted_target else np.zeros(11).tolist())
    voltage_hist = _voltage_histogram(pooled_v, voltage_window)
    angle_objective = (0.5 * (sum(abs(x - 0.10) for x in pooled["unconditional_fractions"][:10])
                              + pooled["unconditional_fractions"][10]) if pooled["finite_count"] else None)
    voltage_bin_count = max(1, len(voltage_hist["counts"]) - 1)
    voltage_objective = (0.5 * (sum(abs(x - 1.0 / voltage_bin_count)
                                     for x in voltage_hist["unconditional_fractions"][:-1])
                                + voltage_hist["unconditional_fractions"][-1])
                         if voltage_hist["finite_count"] else None)
    return {
        "interpretation": "PF-accepted, not equipment-feasible",
        "network_count": len(angle_values),
        "snapshot_count": total_snapshots,
        "branch_count_per_network_total": branch_count,
        "line_count_per_network_total": line_count,
        "transformer_count_per_network_total": transformer_count,
        "branch_definitions_by_network": branch_definitions_by_network,
        "branch_snapshot_observations": pooled,
        "line_branch_snapshot_observations": _target_angle_histogram(
            np.concatenate(line_values) if line_values else np.array([], dtype=float)),
        "transformer_branch_snapshot_observations": _target_angle_histogram(
            np.concatenate(transformer_values) if transformer_values else np.array([], dtype=float)),
        "network_weighted_angle_bin_fractions": net_share_mean,
        "all_bus_solved_v": voltage_hist,
        "pq_bus_solved_v": _voltage_histogram(pooled_pq, voltage_window),
        "missing_or_nonfinite_branch_observations": missing_angles,
        "errors": errors,
        "D_theta": angle_objective,
        "D_V": voltage_objective,
        "D": (0.5 * angle_objective + 0.5 * voltage_objective
              if angle_objective is not None and voltage_objective is not None else None),
        "angle_target_metric": angle_objective,
        "voltage_target_metric": voltage_objective,
    }


def analyze_network(net: Any, system: str = "synthetic", source_index: str = "",
                    index_position: int = 0, relative_path: str = "") -> dict[str, Any]:
    """In-memory analyzable unit used by the CLI and no-solver tests."""
    delta, branches, angle, errors = _branch_observations(net)
    if _get_frame(getattr(net, "buses_t", None), "v_mag_pu").empty:
        # [STSI 081126]:match plot_dataset_distributions, which skips networks without voltage magnitude snapshots.
        errors.append("buses_t.v_mag_pu empty; network excluded from plotted baseline observations")
        delta = np.empty((0, len(branches)), dtype=float)
        angle = angle.iloc[0:0]
        margins = {key: {"source": "unavailable: network excluded because buses_t.v_mag_pu is empty",
                         **_series_stats([])} for key in ("V", "P", "Q")}
        margin_values = {key: np.array([], dtype=float) for key in ("V", "P", "Q")}
    else:
        margins = {}
        margin_values = {}
        for metric, (values, source) in _margin_frames(net).items():
            margin_values[metric] = np.asarray(values, dtype=float).reshape(-1)
            margins[metric] = {"source": source, **_series_stats(values)}
    finite_mask = np.isfinite(delta)
    flat = delta[finite_mask]
    counts = np.bincount(_bin_indices(flat), minlength=len(BIN_LABELS)) if flat.size else np.zeros(len(BIN_LABELS), dtype=int)
    network_fraction = counts / flat.size if flat.size else np.zeros(len(BIN_LABELS), dtype=float)
    branch_type_values = {}
    for kind in ("line", "transformer"):
        mask = (branches.type.to_numpy() == kind) if len(branches) else np.array([], dtype=bool)
        branch_type_values[kind] = delta[:, mask][np.isfinite(delta[:, mask])] if mask.any() else np.array([], dtype=float)
    orientations = []
    if len(branches):
        for (kind, bus0, bus1), count in branches.groupby(["type", "bus0", "bus1"], dropna=False).size().items():
            orientations.append({"type": str(kind), "bus0": str(bus0), "bus1": str(bus1), "count": int(count)})
    orient_set = {(str(r["type"]), str(r["bus0"]), str(r["bus1"])) for r in orientations}
    reversed_pairs = sum(int(r["count"]) for r in orientations
                         if (r["type"], r["bus1"], r["bus0"]) in orient_set)
    maximum = float(np.max(np.abs(flat))) if flat.size else None
    return {"system": system, "source_index": source_index, "index_position": int(index_position),
            "relative_path": relative_path, "snapshot_count": int(len(angle)), "branch_count": int(len(branches)),
            "finite_signed_count": int(flat.size), "nonfinite_signed_count": int(delta.size - flat.size),
            "delta": flat, "bin_counts": counts, "bin_fractions": network_fraction, "max_abs_delta_deg": maximum,
            "signed_stats": _series_stats(flat), "zero_count": int(np.count_nonzero(flat == 0.0)),
            "negative_count": int(np.count_nonzero(flat < 0.0)), "positive_count": int(np.count_nonzero(flat > 0.0)),
            "orientation_counts": orientations, "reversed_pair_occurrences": int(reversed_pairs),
            "line_count": int((branches.type == "line").sum()) if len(branches) else 0,
            "transformer_count": int((branches.type == "transformer").sum()) if len(branches) else 0,
            "margins": margins, "margin_values": margin_values, "branch_type_values": branch_type_values, "errors": errors,
            "available": {"angles": bool(flat.size), "margins": {k: v["count"] > 0 for k, v in margins.items()}}}


def _sidecar_summary(sidecar: dict[str, Any] | None, indexed_files: list[str], system: str) -> dict[str, Any]:
    if sidecar is None:
        return {"available": False, "error": "sidecar missing/unreadable", "reconciled": False}
    rows = sidecar.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    indexed_positions = {i for i, _ in enumerate(indexed_files)}
    accepted_positions = []
    for row in rows:
        if isinstance(row, dict) and row.get("status") in ("accepted_full", "accepted_partial"):
            if isinstance(row.get("file_position"), int):
                accepted_positions.append(row["file_position"])
    duplicate_positions = sorted(p for p, n in Counter(accepted_positions).items() if n > 1)
    out_of_range = sorted(p for p in set(accepted_positions) if p not in indexed_positions)
    expected_accepted = sidecar.get("n_accepted")
    position_reconciled = (accepted_positions == list(range(len(indexed_files))) and not duplicate_positions and not out_of_range
                           and (expected_accepted is None or expected_accepted == len(accepted_positions)))
    status_counts = Counter(row.get("status", "missing_status") for row in rows if isinstance(row, dict))
    accepted_rows = [r for r in rows if isinstance(r, dict) and r.get("status") in ("accepted_full", "accepted_partial")]
    def evaluated_count(row: dict[str, Any]) -> int:
        if row.get("status") in ("accepted_full", "accepted_partial", "rejected_snapshots"):
            return int(row.get("n_snapshots") or 0)
        return 0

    by_bucket: dict[str, dict[str, Any]] = {}
    row_reasons: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        draw = row.get("draws") or {}
        bucket = draw.get("lam_bucket")
        key = str(bucket) if bucket is not None else "missing-draw"
        entry = by_bucket.setdefault(key, {"attempts": 0, "statuses": {}, "evaluated_snapshots": 0,
                                           "accepted_snapshots": 0, "reasons": Counter(),
                                           "recorded_attempt_reasons": Counter()})
        entry["attempts"] += 1
        entry["statuses"][row.get("status", "missing_status")] = entry["statuses"].get(row.get("status", "missing_status"), 0) + 1
        if row.get("reason") is not None:
            entry["recorded_attempt_reasons"][str(row["reason"])] += 1
        entry["declared_n_snapshots"] = entry.get("declared_n_snapshots", 0) + int(row.get("n_snapshots") or 0)
        entry["evaluated_snapshots"] += evaluated_count(row)
        entry["accepted_snapshots"] += int(row.get("n_ok") or 0) if row.get("status") in ("accepted_full", "accepted_partial") else 0
        for failed_key, value in (row.get("failed") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry["reasons"][str(failed_key)] += int(value)
                row_reasons[str(failed_key)] += int(value)
            elif isinstance(value, str):
                # [STSI 081126]:sidecar failed maps may be snapshot->first-rejection reason rather than reason->count.
                entry["reasons"][value] += 1
                row_reasons[value] += 1
            else:
                entry["reasons"][str(failed_key)] += 1
                row_reasons[str(failed_key)] += 1
        if row.get("status") in ("accepted_full", "accepted_partial"):
            ok_count = int(row.get("n_ok") or 0)
            row_reasons["ok"] += ok_count
            entry["reasons"]["ok"] += ok_count
    for entry in by_bucket.values():
        entry["reasons"] = dict(entry["reasons"])
        entry["recorded_attempt_reasons"] = dict(entry["recorded_attempt_reasons"])
    reason_aggregate = sidecar.get("snapshot_reason_counts")
    reason_reconciled = None
    if isinstance(reason_aggregate, dict):
        reason_reconciled = all(int(reason_aggregate.get(k, 0) or 0) == row_reasons.get(k, 0)
                                for k in set(reason_aggregate) | set(row_reasons))
    attempt_rows = [{key: row.get(key) for key in
                     ("attempt", "status", "reason", "n_snapshots", "n_ok", "failed", "draws", "file_position", "max_dtheta_deg")}
                    for row in rows if isinstance(row, dict)]
    attempts_reconciled = sidecar.get("n_attempts") in (None, len(rows))
    accepted_statuses = status_counts.get("accepted_full", 0) + status_counts.get("accepted_partial", 0)
    accepted_reconciled = expected_accepted in (None, accepted_statuses)
    reconciled = position_reconciled and attempts_reconciled and accepted_reconciled and reason_reconciled is not False
    return {"available": True, "n_attempts": sidecar.get("n_attempts"),
            "target_networks": sidecar.get("num_scenarios"), "n_accepted": expected_accepted,
            "partial": sidecar.get("partial"), "seed": sidecar.get("seed"), "config": sidecar.get("config"),
            "generator_args": sidecar.get("generator_args"), "steering": sidecar.get("steering"),
            "wall_time_s": sidecar.get("wall_time_s"),
            "lambda_ranges_verifiable": any(k in sidecar for k in ("lambda_range", "lam_range", "lambda_ranges")) or
                                        any(k in (sidecar.get("config") or {}) for k in ("lambda_range", "lam_range", "lambda_ranges")),
            "calibration_verifiable": "calibration" in (sidecar.get("config") or {}),
            "rejection_verifiable": "acceptance" in sidecar or "snapshot_reason_counts" in sidecar,
            "steering_verifiable": "steering" in sidecar or "steering" in (sidecar.get("config") or {}),
            "snapshot_reason_counts": sidecar.get("snapshot_reason_counts"),
            "scenario_status_counts": sidecar.get("scenario_status_counts"),
            "rows_present": len(rows), "row_status_counts": dict(status_counts),
            "declared_n_snapshots": sum(int(r.get("n_snapshots") or 0) for r in rows if isinstance(r, dict)),
            "evaluated_snapshots": sum(evaluated_count(r) for r in rows if isinstance(r, dict)),
            "evaluation_count_unknown_attempts": sum(1 for r in rows if isinstance(r, dict)
                                                      and r.get("status") == "exception" and r.get("n_snapshots") is None),
            "accepted_snapshots": sum(int(r.get("n_ok") or 0) for r in accepted_rows),
            "accepted_full": status_counts.get("accepted_full", 0), "accepted_partial": status_counts.get("accepted_partial", 0),
            "structural_rejects": status_counts.get("rejected_structural", 0),
            "exceptions": status_counts.get("exception", 0), "accepted_file_positions": accepted_positions,
            "attempt_rows": attempt_rows, "row_snapshot_reason_counts": dict(row_reasons),
            "snapshot_reason_reconciliation": reason_reconciled,
            "attempt_reconciliation": {"rows_match_n_attempts": attempts_reconciled,
                                       "status_accepted_matches_n_accepted": accepted_reconciled},
            "duplicate_positions": duplicate_positions, "out_of_range_positions": out_of_range,
            "position_reconciliation": {"indexed_networks": len(indexed_files), "accepted_rows_with_position": len(accepted_positions),
                                        "positions_match": position_reconciled},
            "lambda_bucket_rows": by_bucket,
            "reconciled": reconciled,
            "rejected_signed_bin_attribution": "unavailable: sidecar stores accepted-snapshot maximum only; failed signed branch values are not linked"}


def _system_template(system: str) -> dict[str, Any]:
    return {"system": system, "status": "not selected", "index": None, "sidecar": None, "manifest": [],
            "errors": [], "network_summaries": [], "summary": {"network_count": 0, "usable_network_count": 0,
            "snapshot_count": 0, "branch_count": 0, "finite_branch_snapshots": 0}}


def _analyze_system(system: str, index_path: Path, notebook_identity: dict[str, Any],
                    versions: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _system_template(system)
    result["status"] = "incomplete"
    result["index"] = _file_identity(index_path)
    result["notebook"] = notebook_identity
    result["versions"] = versions
    network_rows: list[dict[str, Any]] = []
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        file_names = index_data.get("files")
        if not isinstance(file_names, list):
            raise ValueError("index 'files' must be a list")
    except Exception as exc:
        result["errors"].append(f"index read/parse error: {type(exc).__name__}: {exc}")
        result["sidecar"] = {"available": False, "error": "index unreadable"}
        return result, network_rows
    result["index_metadata"] = {k: v for k, v in index_data.items() if k != "files"}
    duplicates = sorted(name for name, count in Counter(map(str, file_names)).items() if count > 1)
    if duplicates:
        result["errors"].append(f"index contains duplicate relative network paths: {duplicates!r}")
    sidecar_path = index_path.with_name(index_path.stem + "_meta.json")
    result["sidecar_identity"] = _file_identity(sidecar_path)
    try:
        sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        sidecar_payload = None
        result["errors"].append(f"sidecar read/parse error: {type(exc).__name__}: {exc}")
    result["sidecar"] = _sidecar_summary(sidecar_payload, file_names, system)
    if sidecar_payload:
        result["sidecar"]["index_sidecar_identity"] = _file_identity(sidecar_path)
        conflicts = []
        if sidecar_payload.get("tag") is not None and index_data.get("tag") is not None and sidecar_payload.get("tag") != index_data.get("tag"):
            conflicts.append(f"sidecar tag {sidecar_payload.get('tag')!r} differs from index tag {index_data.get('tag')!r}")
        if sidecar_payload.get("base_system") is not None and sidecar_payload.get("base_system") != system:
            conflicts.append(f"sidecar base_system {sidecar_payload.get('base_system')!r} differs from selected system {system!r}")
        result["sidecar"]["identity_conflicts"] = conflicts
        if conflicts:
            result["sidecar"]["reconciled"] = False
            result["errors"].extend(conflicts)
        cfg = sidecar_payload.get("config")
        result["sidecar"]["config_provenance_note"] = (
            "Recorded in sidecar; no settings inferred from the current notebook." if isinstance(cfg, dict)
            else "Configuration unavailable from sidecar.")
    if not file_names:
        result["errors"].append("index contains no networks")
    # Load PyPSA lazily only when an indexed network exists; never invoke pf or generation.
    try:
        import pypsa
        pypsa_version = getattr(pypsa, "__version__", "unknown")
        result["versions"]["pypsa"] = str(pypsa_version)
    except Exception as exc:
        pypsa = None
        result["errors"].append(f"PyPSA import error: {type(exc).__name__}: {exc}")
    if pypsa is None:
        for position, name in enumerate(file_names):
            rel = str(name)
            path = index_path.parent / rel
            result["manifest"].append({"position": position, "relative_path": rel, **_file_identity(path)})
            network_rows.append(_failed_network_row(system, str(index_path), position, rel, "PyPSA unavailable"))
        return result, network_rows
    for position, name in enumerate(file_names):
        relative = str(name)
        path = index_path.parent / relative
        identity = _file_identity(path)
        manifest_entry = {"position": position, "relative_path": relative, **identity}
        result["manifest"].append(manifest_entry)
        if identity["error"]:
            result["errors"].append(f"network[{position}] {relative}: {identity['error']}")
            network_rows.append(_failed_network_row(system, str(index_path), position, relative, identity["error"]))
            continue
        try:
            network = pypsa.Network(str(path))
            record = analyze_network(network, system, str(index_path), position, relative)
            record["availability_flags"] = record["available"]
            network_rows.append(record)
            if record["errors"]:
                result["errors"].extend(f"network[{position}] {relative}: {e}" for e in record["errors"])
            if record["nonfinite_signed_count"]:
                result["errors"].append(f"network[{position}] {relative}: {record['nonfinite_signed_count']} nonfinite/missing signed values of {record['snapshot_count'] * record['branch_count']}")
            if not record["snapshot_count"] or not record["branch_count"]:
                result["errors"].append(f"network[{position}] {relative}: zero snapshots or zero branches")
            for metric, info in record["margins"].items():
                if info["missing_or_nonfinite"]:
                    result["errors"].append(f"network[{position}] {relative}: {metric} has {info['missing_or_nonfinite']} missing/nonfinite values")
                if not info["count"]:
                    result["errors"].append(f"network[{position}] {relative}: no finite {metric} margin observations")
            del network
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"network[{position}] {relative}: {error}")
            network_rows.append(_failed_network_row(system, str(index_path), position, relative, error))
    result["network_summaries"] = [{k: v for k, v in row.items() if k not in
                                    ("delta", "bin_counts", "bin_fractions", "margin_values", "branch_type_values",
                                     "orientation_counts")}
                                   for row in network_rows]
    valid = [r for r in network_rows if r.get("finite_signed_count", 0) > 0]
    result["summary"] = _summarize_system(system, network_rows, valid)
    if not result["sidecar"].get("available"):
        result["errors"].append("generation sidecar unavailable")
    elif not result["sidecar"].get("reconciled"):
        result["errors"].append("generation sidecar accepted file positions do not reconcile to the index")
    result["status"] = "complete" if file_names and len(result["manifest"]) == len(file_names) and not result["errors"] else "incomplete"
    result["provenance"] = {"index_tag": index_data.get("tag"), "index_timestamp": index_data.get("timestamp"),
                            "candidate_baseline_source_unverified": True,
                            "comparison_verified": bool(result["sidecar"].get("reconciled")) and
                                                   not result["index"].get("error") and not result["sidecar_identity"].get("error")}
    return result, network_rows


def _failed_network_row(system: str, source_index: str, position: int, relative: str, error: str) -> dict[str, Any]:
    return {"system": system, "source_index": source_index, "index_position": position, "relative_path": relative,
            "snapshot_count": None, "branch_count": None, "finite_signed_count": 0, "nonfinite_signed_count": None,
            "bin_counts": np.zeros(len(BIN_LABELS), dtype=int), "bin_fractions": np.zeros(len(BIN_LABELS)),
            "max_abs_delta_deg": None, "margins": {}, "errors": [error],
            "available": {"angles": False, "margins": {"V": False, "P": False, "Q": False}}}


def _summarize_system(system: str, rows: list[dict[str, Any]], valid: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = np.concatenate([r["delta"] for r in valid]) if valid else np.array([], dtype=float)
    counts = np.sum([r["bin_counts"] for r in valid], axis=0).astype(int) if valid else np.zeros(len(BIN_LABELS), dtype=int)
    max_rows = [r for r in valid if r["max_abs_delta_deg"] is not None]
    maxima = np.asarray([r["max_abs_delta_deg"] for r in max_rows], dtype=float)
    cohorts = {str(i): [] for i in range(4)}
    thresholds = np.quantile(maxima, [0.25, 0.5, 0.75], method="linear").tolist() if maxima.size else []
    if maxima.size:
        for row, maximum in zip(max_rows, maxima):
            cohorts[str(int(np.searchsorted(thresholds, maximum, side="left")))].append(row)
    per_network_min_median_max = {}
    for i, label in enumerate(BIN_LABELS):
        fracs = [float(r["bin_fractions"][i]) for r in valid]
        per_network_min_median_max[label] = ({"min": min(fracs), "median": float(np.median(fracs)), "max": max(fracs)}
                                             if fracs else {"min": None, "median": None, "max": None})
    qlo, qhi = np.percentile(pooled, [1, 99]).tolist() if pooled.size else (None, None)
    trimmed = pooled[(pooled >= qlo) & (pooled <= qhi)] if pooled.size else np.array([])
    total = int(pooled.size)
    top = cohorts["3"]
    top_count = int(sum(r["finite_signed_count"] for r in top))
    near_count = int(sum(np.count_nonzero((r["delta"] >= -5.0) & (r["delta"] < 5.0)) for r in top))
    margins = {}
    top_ids = {id(r) for r in top}
    for metric in ("V", "P", "Q"):
        all_v, top_v, sources = [], [], set()
        top_ids = {id(r) for r in top}
        for r in rows:
            m = r.get("margins", {}).get(metric, {})
            if m.get("count"):
                all_v.extend(r.get("margin_values", {}).get(metric, np.array([])).tolist())
                if id(r) in top_ids:
                    top_v.extend(r.get("margin_values", {}).get(metric, np.array([])).tolist())
                sources.add(m.get("source"))
        all_stats = _series_stats(all_v)
        top_stats = _series_stats(top_v)
        margins[metric] = {"network_count": sum(bool(r.get("margins", {}).get(metric, {}).get("count")) for r in rows),
                           "top_cohort_network_count": sum(bool(r.get("margins", {}).get(metric, {}).get("count")) for r in top),
                           "sources": sorted(str(s) for s in sources),
                           "pooled_bus_snapshot_weighting": all_stats,
                           "top_cohort_pooled_bus_snapshot_weighting": top_stats,
                           "median_shift_top_minus_all": (top_stats["quantiles"]["50"] - all_stats["quantiles"]["50"])
                           if top_v and all_v else None,
                           "fallback_or_missingness_differs": len(sources) > 1 or
                           sum(bool(r.get("margins", {}).get(metric, {}).get("count")) for r in rows) < len(rows)}
    branch_types = {}
    for kind in ("line", "transformer"):
        values = np.concatenate([r.get("branch_type_values", {}).get(kind, np.array([])) for r in valid]) if valid else np.array([])
        branch_types[kind] = {"count": int(values.size), "statistics": _series_stats(values),
                              "bin_counts": np.bincount(_bin_indices(values), minlength=len(BIN_LABELS)).tolist()
                              if values.size else [0] * len(BIN_LABELS)}
    return {"network_count": len(rows), "usable_network_count": len(valid),
            "snapshot_count": int(sum(r.get("snapshot_count") or 0 for r in rows)),
            "branch_count": int(sum(r.get("branch_count") or 0 for r in rows)),
            "finite_branch_snapshots": total,
            "raw_signed_statistics": _series_stats(pooled), "trimmed_1_99_count": int(trimmed.size),
            "trimmed_1_99_statistics": _series_stats(trimmed), "bin_counts": counts.tolist(),
            "pooled_bin_fractions": (counts / total).tolist() if total else [None] * len(BIN_LABELS),
            "bin_contributing_networks": [int(sum(r["bin_counts"][i] > 0 for r in valid)) for i in range(len(BIN_LABELS))],
            "negative_count": int(np.count_nonzero(pooled < 0)), "exact_zero_count": int(np.count_nonzero(pooled == 0)),
            "positive_count": int(np.count_nonzero(pooled > 0)),
            "nonempty_bin_intervals": [label for label, count in zip(BIN_LABELS, counts) if count],
            "per_network_bin_fraction_min_median_max": per_network_min_median_max,
            "network_max_abs_delta_statistics": _series_stats(maxima),
            "cohort_thresholds_q25_q50_q75": thresholds,
            "cohort_sizes": {key: len(value) for key, value in cohorts.items()},
            "top_cohort": {"available": bool(top), "network_count": len(top), "finite_branch_snapshots": top_count,
                           "bin_counts": (np.sum([r["bin_counts"] for r in top], axis=0).astype(int).tolist() if top
                                          else [0] * len(BIN_LABELS)),
                           "bin_fractions": ((np.sum([r["bin_counts"] for r in top], axis=0) / top_count).tolist()
                                             if top_count else [None] * len(BIN_LABELS)),
                           "fraction_in_minus5_to_plus5": near_count / top_count if top_count else None},
            "line_count": int(sum(r.get("line_count", 0) for r in valid)),
            "transformer_count": int(sum(r.get("transformer_count", 0) for r in valid)),
            "reversed_pair_occurrences": int(sum(r.get("reversed_pair_occurrences", 0) for r in valid)),
            "orientation_counts": _aggregate_orientations(valid),
            "margin_network_counts": margins, "line_transformer_signed": branch_types}


def _aggregate_orientations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        for item in row.get("orientation_counts", []):
            counts[(item["type"], item["bus0"], item["bus1"])] += int(item["count"])
    return [{"type": kind, "bus0": bus0, "bus1": bus1, "count": count}
            for (kind, bus0, bus1), count in sorted(counts.items())]


def _write_outputs(output_dir: Path, data_root: Path, selected: dict[str, Path], overrides_used: bool) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    run_started = datetime.now(timezone.utc)
    notebook_identity = _file_identity(NOTEBOOK_PATH)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pypsa": "not loaded"}
    try:
        import pypsa
        versions["pypsa"] = str(getattr(pypsa, "__version__", "unknown"))
    except Exception as exc:
        versions["pypsa"] = f"unavailable: {type(exc).__name__}: {exc}"
    systems: dict[str, Any] = {}
    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    # Build full six-system output even if source indexes are absent or not selected.
    for system in SYSTEMS:
        if system not in selected:
            systems[system] = _system_template(system)
            systems[system]["reason"] = "not selected because explicit --index overrides were supplied" if overrides_used else "default candidate index not found"
            systems[system]["notebook"] = notebook_identity
            systems[system]["versions"] = versions.copy()
            rows_by_system[system] = []
            if not overrides_used:
                systems[system]["status"] = "incomplete"
                default_index = data_root / "training_networks_saved" / INDEX_NAMES[system]
                if not default_index.exists():
                    systems[system]["index"] = _file_identity(default_index)
                    systems[system]["errors"] = [f"candidate index missing: {default_index}"]
            continue
        systems[system], rows_by_system[system] = _analyze_system(system, selected[system], notebook_identity, versions.copy())
    # Pooled and equally weighted per-network signed bins, including explicit under/overflow.
    bin_records = []
    stats_records = []
    for system in SYSTEMS:
        valid = [r for r in rows_by_system[system] if r.get("finite_signed_count", 0) > 0]
        pooled_count = int(sum(r["finite_signed_count"] for r in valid))
        total_networks = len(valid)
        for idx, label in enumerate(BIN_LABELS):
            count = int(sum(r["bin_counts"][idx] for r in valid))
            fractions = [float(r["bin_fractions"][idx]) for r in valid]
            bin_records.append({"system": system, "bin_index": idx, "bin_label": label,
                                "lower_edge_deg": "" if idx == 0 else (-80 if idx == 1 else float(BIN_EDGES[idx - 1])),
                                "upper_edge_deg": "" if idx == len(BIN_LABELS)-1 else (-80 if idx == 0 else float(BIN_EDGES[idx])),
                                "pooled_branch_snapshot_count": count,
                                "pooled_fraction": count / pooled_count if pooled_count else "",
                                "contributing_networks": sum(r["bin_counts"][idx] > 0 for r in valid),
                                "mean_per_network_fraction": float(np.mean(fractions)) if fractions else ""})
        summary = systems[system].get("summary", {})
        thresholds = summary.get("cohort_thresholds_q25_q50_q75", [])
        cohorts = {str(i): [] for i in range(4)}
        max_rows = [r for r in valid if r.get("max_abs_delta_deg") is not None]
        for r in max_rows:
            cohort = int(np.searchsorted(thresholds, r["max_abs_delta_deg"], side="left")) if thresholds else 0
            cohorts[str(cohort)].append(r)
        for r in rows_by_system[system]:
            rec = {"system": system, "source_index": r.get("source_index", ""),
                   "index_position": r.get("index_position", ""), "relative_path": r.get("relative_path", ""),
                   "kept_snapshot_count": r.get("snapshot_count", ""), "branch_count": r.get("branch_count", ""),
                   "finite_signed_count": r.get("finite_signed_count", 0),
                   "nonfinite_signed_count": r.get("nonfinite_signed_count", ""),
                   "max_abs_delta_deg": r.get("max_abs_delta_deg", ""),
                   "angles_available": bool(r.get("available", {}).get("angles")),
                   "V_available": bool(r.get("available", {}).get("margins", {}).get("V", False)),
                   "P_available": bool(r.get("available", {}).get("margins", {}).get("P", False)),
                   "Q_available": bool(r.get("available", {}).get("margins", {}).get("Q", False))}
            rec.update({f"count_{i}": int(r.get("bin_counts", np.zeros(len(BIN_LABELS)))[i]) for i in range(len(BIN_LABELS))})
            rec.update({f"fraction_{i}": float(r.get("bin_fractions", np.zeros(len(BIN_LABELS)))[i]) for i in range(len(BIN_LABELS))})
            stats_records.append(rec)
    # Attach coherent per-system sidecar yield tables and exact manifest totals.
    for system, records in rows_by_system.items():
        s = systems[system]
        s.setdefault("summary", _summarize_system(system, [], []))
        s["summary"].setdefault("network_count", len(records))
        s["manifest_summary"] = {"indexed_networks": len(s.get("manifest", [])),
                                 "hashed_networks": sum(m.get("sha256") is not None for m in s.get("manifest", [])),
                                 "network_bytes": sum(m.get("size_bytes") or 0 for m in s.get("manifest", [])),
                                 "network_load_errors": len([r for r in records if r.get("errors")])}
        s["sidecar_yield"] = s.get("sidecar") if isinstance(s.get("sidecar"), dict) and s["sidecar"].get("available") else None
        # [STSI 081126]:preserve pooled bus-snapshot summaries and source/fallback details for each margin.
        s["summary"]["margins"] = {
            metric: {"units": "p.u. for V; as stored by PyPSA for P/Q",
                     **s["summary"].get("margin_network_counts", {}).get(metric, {})}
            for metric in ("V", "P", "Q")}
        s["summary"]["sidecar_presence"] = bool(isinstance(s.get("sidecar"), dict) and s["sidecar"].get("available"))
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_obj = {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "run_started_utc": run_started.isoformat(), "analysis_completed_utc": datetime.now(timezone.utc).isoformat(),
                   "data_root": str(data_root.resolve(strict=False)),
                   "index_overrides_used": overrides_used, "selected_systems": list(selected), "notebook": notebook_identity,
                   "versions": versions, "bin_labels": BIN_LABELS, "bin_edges_deg": BIN_EDGES.tolist(),
                   "rejected_signed_bin_attribution": "unavailable unless failed snapshot branch-angle values are stored and linkable",
                   "systems": systems,
                   "manifest": {system: systems[system].get("manifest", []) for system in SYSTEMS},
                   "outputs": list(OUTPUT_FILES)}
    (output_dir / "summary.json").write_text(json.dumps(summary_obj, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    with (output_dir / "signed_bins.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bin_records[0]) if bin_records else ["system"])
        writer.writeheader(); writer.writerows(bin_records)
    with (output_dir / "network_stats.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(stats_records[0]) if stats_records else ["system"])
        writer.writeheader(); writer.writerows(stats_records)
    report = _render_report(summary_obj)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary_obj


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "": return "unavailable"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}" if math.isfinite(float(value)) else "unavailable"
    return str(value)


def _render_report(summary: dict[str, Any]) -> str:
    lines = ["# Signed branch-angle baseline feasibility diagnostics", "",
             f"- Created UTC: {summary['created_utc']}", f"- Data root: `{summary['data_root']}`",
             f"- Notebook SHA-256: `{summary['notebook'].get('sha256') or 'unavailable'}`",
             "- Candidate datasets are a baseline candidate only; their relationship to the original figure is unverified.",
             "- Read-only analysis: no PF, generation, calibration, steering, or source mutation was performed.", "",
             "## Six-system evidence gate", "",
             "| System | Status | Indexed / hashed | Networks / snapshots / branches / finite Δθ | Raw signed support | Tails | Top cohort | V/P/Q | Yield / gaps |",
             "|---|---|---:|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        s = summary["systems"][system]
        sm = s.get("summary", {})
        raw = sm.get("raw_signed_statistics", {})
        n = s.get("manifest_summary", {})
        top = sm.get("top_cohort", {})
        side = s.get("sidecar") or {}
        margins = sm.get("margins", {})
        margin_txt = "/".join(str(margins.get(k, {}).get("network_count", 0)) for k in ("V", "P", "Q"))
        side_yield = "verified" if side.get("reconciled") else "unverified"
        tail_counts = sm.get("bin_counts", [0] * len(BIN_LABELS))
        contributors = sm.get("bin_contributing_networks", [0] * len(BIN_LABELS))
        lines.append(f"| {system} | {s.get('status')} | {n.get('indexed_networks',0)} / {n.get('hashed_networks',0)} | "
                     f"{sm.get('usable_network_count',0)} / {sm.get('snapshot_count',0)} / {sm.get('branch_count',0)} / "
                     f"{sm.get('finite_branch_snapshots',0)} | {_fmt(raw.get('min'))}…{_fmt(raw.get('max'))}; "
                     f"{len(sm.get('nonempty_bin_intervals', []))} bins | "
                     f"{tail_counts[0]}/{contributors[0]} below; {tail_counts[-1]}/{contributors[-1]} above | "
                     f"{top.get('network_count',0)} n; ±5°={_fmt(top.get('fraction_in_minus5_to_plus5'))} | "
                     f"{margin_txt} networks V/P/Q | {side.get('n_attempts','unavailable')} attempts; {side_yield}; "
                     f"{len(s.get('errors', []))} errors |")
    lines += ["", "## Definitions and interpretation", "",
              "The signed observable is `degrees(v_ang[bus0] - v_ang[bus1])`, with no absolute value, wrap, or transformer phase correction. Lines and transformers are concatenated as separate branch rows; parallel circuits remain separate samples. Swapping bus0/bus1 negates Δθ, so observed sign imbalance partly reflects stored orientation conventions.",
              "Bins are half-open [−80,−75), …, [75,80), with below −80 and ≥80 tails. Every finite value occurs once in pooled counts; exact zero is separately counted and falls in [0,5) under the histogram convention. The 1st–99th-percentile trimmed summaries match the plot trim only; raw extrema and the full histogram are untrimmed.",
              "Per-network bin-fraction minima/medians/maxima describe bounds on convex mixtures of the observed network histograms only. They are not jointly attainable target vectors or a proof of generator feasibility. A top-quartile network is selected by the prescribed maximum-|Δθ| quantile partition; its branch samples still span all bins and are not an independent-branch target.",
              "Snapshot and branch-snapshot observations are correlated within networks; they are descriptive counts, not independent trials. No missing or inaccessible observation is interpreted as zero coverage.", "",
              "## Per-system evidence and provenance", ""]
    for system in SYSTEMS:
        s = summary["systems"][system]; sm = s.get("summary", {})
        lines += [f"### {system}: {s.get('status')}", "",
                  f"- Index: `{(s.get('index') or {}).get('path')}`; SHA-256 `{(s.get('index') or {}).get('sha256')}`",
                  f"- Sidecar: `{(s.get('sidecar_identity') or {}).get('path')}`; SHA-256 `{(s.get('sidecar_identity') or {}).get('sha256')}`",
                  f"- Candidate config: seed `{(s.get('sidecar') or {}).get('seed')}`, partial `{(s.get('sidecar') or {}).get('partial')}`, config recorded `{(s.get('sidecar') or {}).get('config') is not None}`; λ range verified `{(s.get('sidecar') or {}).get('lambda_ranges_verifiable')}`, calibration `{(s.get('sidecar') or {}).get('calibration_verifiable')}`, rejection `{(s.get('sidecar') or {}).get('rejection_verifiable')}`, steering `{(s.get('sidecar') or {}).get('steering_verifiable')}`; comparison verified `{s.get('provenance', {}).get('comparison_verified')}`.",
                  f"- Counts: {sm.get('network_count',0)} indexed entries; {sm.get('usable_network_count',0)} usable networks; {sm.get('snapshot_count',0)} kept snapshots; {sm.get('branch_count',0)} branches; {sm.get('finite_branch_snapshots',0)} finite Δθ values.",
                  f"- Signed raw extrema `{_fmt(sm.get('raw_signed_statistics',{}).get('min'))}` to `{_fmt(sm.get('raw_signed_statistics',{}).get('max'))}`; nonempty intervals: {', '.join(sm.get('nonempty_bin_intervals',[])) or 'none'}; negative/zero/positive {sm.get('negative_count',0)}/{sm.get('exact_zero_count',0)}/{sm.get('positive_count',0)}.",
                  f"- Raw 1/5/25/50/75/95/99% quantiles `{sm.get('raw_signed_statistics',{}).get('quantiles',{})}`; 1–99% trimmed count/statistics `{sm.get('trimmed_1_99_count',0)}` / `{sm.get('trimmed_1_99_statistics',{})}`.",
                  f"- Tails: below −80° {sm.get('bin_counts',[0]*len(BIN_LABELS))[0]} from {sm.get('bin_contributing_networks',[0]*len(BIN_LABELS))[0]} networks; ≥80° {sm.get('bin_counts',[0]*len(BIN_LABELS))[-1]} from {sm.get('bin_contributing_networks',[0]*len(BIN_LABELS))[-1]} networks; network maxima summary `{sm.get('network_max_abs_delta_statistics',{})}`.",
                  f"- Cohorts: thresholds Q25/Q50/Q75 `{sm.get('cohort_thresholds_q25_q50_q75',[])}`; sizes `{sm.get('cohort_sizes',{})}`; top cohort {sm.get('top_cohort',{})}.",
                  f"- Line/transformer branch counts {sm.get('line_count',0)}/{sm.get('transformer_count',0)}; reversed-pair occurrences {sm.get('reversed_pair_occurrences',0)}.",
                  f"- Line/transformer signed distributions `{sm.get('line_transformer_signed',{})}`; observed orientation tuples `{sm.get('orientation_counts',[])}`.",
                  f"- Margins: `{sm.get('margins',{})}`. Units are stored PyPSA units (V in p.u.); fallbacks/source are identified per-network in network_stats availability and summary manifest.",
                  f"- Sidecar outcome: attempts `{(s.get('sidecar') or {}).get('n_attempts')}`, target `{(s.get('sidecar') or {}).get('target_networks')}`, accepted `{(s.get('sidecar') or {}).get('n_accepted')}`, statuses `{(s.get('sidecar') or {}).get('row_status_counts')}`, evaluated snapshots `{(s.get('sidecar') or {}).get('evaluated_snapshots')}`, accepted snapshots `{(s.get('sidecar') or {}).get('accepted_snapshots')}`; reconciled `{(s.get('sidecar') or {}).get('reconciled')}`; historical sidecar wall time `{(s.get('sidecar') or {}).get('wall_time_s')}` seconds; source network sizes `{s.get('manifest_summary', {}).get('network_bytes')}` bytes. Future pilot runtime/disk estimate is unknown from this read-only diagnostic; use available historical measures only.",
                  f"- Missingness/errors: `{s.get('errors',[])}`", ""]
    lines += ["## Recorded rejection evidence", "",
              "Sidecar attempts, accepted statuses, structural rejects, exceptions, evaluated/accepted snapshots, first/dominant-rule reason counts, and λ-bucket rows are retained in `summary.json`. Accepted file positions are checked against index positions. Sidecar `max_dtheta_deg` concerns accepted snapshots only. Rejected signed-bin attribution is unavailable unless rejected branch-angle measurements are recorded and linkable; no rejected attempt is assigned to a signed bin.", "",
              "## Limitations and human gate", "",
              "This diagnostic does not establish feasibility, reachability, physical boundaries, or a target. Missing indexes, sidecars, files, nonfinite values, and provenance conflicts are gaps, not evidence of zero coverage. The per-system `comparison_verified` flag is withheld when sidecar/index reconciliation is unavailable or conflicting.",
              "A human must select observed working signed support and desired mass per system and specify tolerable V/P/Q shifts. This report recommends no numerical steering target and authorizes no steering, calibration, simulation, or pilot.",
              "If evidence is insufficient, a separately approved exploratory pilot may request at most 100 accepted networks per affected system, seed 42, 24 snapshots, and at most 200 attempts per system (600 requested / 1,200 attempts if all six need it), using current saved notebook UC_COMMON/UC_SYSTEMS settings, calibration off, fixed λ/κ ranges, and unchanged rejection rules. Use unique tags `dtheta_diag_<system>_100_20261108` in an isolated approved data root. No tuning or second round. Pilot is exploratory, not proof of bounds or a representative 1500-network replacement. Before any pilot, obtain separate explicit approval of systems, budget/runtime/disk and destinations; verify `.venv`, grid files, effective config, and absence of matching tags/index/subfolder/sidecar/coverage report. Do not execute if any prerequisite fails. Estimate runtime/disk from existing sidecar wall times and file sizes; if unavailable, report unknown.", "",
              "## Output files", ""]
    lines.extend(f"- `{name}`" for name in OUTPUT_FILES)
    lines.append("")
    return "\n".join(lines)


def _parse_index_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("override must be SYSTEM=ABSOLUTE_PATH")
    system, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if system not in SYSTEMS:
        raise argparse.ArgumentTypeError(f"unknown system {system!r}; expected one of {', '.join(SYSTEMS)}")
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"index path must be absolute: {raw_path!r}")
    return system, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", action="append", default=[], metavar="SYSTEM=ABSOLUTE_PATH",
                        help="repeatable; when used, only named systems are selected")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected: dict[str, Path] = {}
    if args.index:
        for raw in args.index:
            system, path = _parse_index_override(raw)
            if system in selected:
                raise SystemExit(f"duplicate --index system: {system}")
            selected[system] = path
    else:
        network_dir = args.data_root / "training_networks_saved"
        for system, name in INDEX_NAMES.items():
            path = network_dir / name
            if path.exists():
                selected[system] = path
    try:
        result = _write_outputs(args.output_dir, args.data_root, selected, bool(args.index))
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Diagnostic report: {args.output_dir.resolve() / 'report.md'}")
    for system in SYSTEMS:
        item = result["systems"][system]
        sm = item.get("summary", {})
        print(f"{system}: {item.get('status')}; networks={sm.get('usable_network_count', 0)}; "
              f"snapshots={sm.get('snapshot_count', 0)}; finite_branch_snapshots={sm.get('finite_branch_snapshots', 0)}; "
              f"errors={len(item.get('errors', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
