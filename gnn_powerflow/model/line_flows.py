"""Line flow calculation from voltage predictions.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 14 /
        GNN_Powerflow_V2.7_Analysis.ipynb (line flows cell)
TODO: remove duplicate inline definitions from notebooks once refactored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa


def calculate_line_flows(
    network: pypsa.Network,
    vmag: np.ndarray,
    vang: np.ndarray,
    t_idx: int | None = None,
) -> dict:
    """
    Calculate AC line flows from predicted voltages using the π model.

    Args:
        network: PyPSA network object
        vmag:    Voltage magnitudes [num_buses], numpy array, p.u.
        vang:    Voltage angles [num_buses], numpy array, radians
        t_idx:   Snapshot index (unused, kept for API compat)

    Returns:
        dict with keys p0, p1, q0, q1 — each a numpy array of length num_lines (p.u.)
    """
    num_lines = len(network.lines)
    p0 = np.zeros(num_lines)
    p1 = np.zeros(num_lines)
    q0 = np.zeros(num_lines)
    q1 = np.zeros(num_lines)

    all_buses = list(network.buses.index)
    bus_to_idx = {bus_name: idx for idx, bus_name in enumerate(all_buses)}

    for i, line in enumerate(network.lines.index):
        from_bus = network.lines.loc[line, "bus0"]
        to_bus = network.lines.loc[line, "bus1"]
        from_idx = bus_to_idx[from_bus]
        to_idx = bus_to_idx[to_bus]

        r = network.lines.loc[line, "r"]
        x = network.lines.loc[line, "x"]
        b = network.lines.loc[line, "b"] if "b" in network.lines.columns else 0.0

        z = complex(r, x)
        y_series = 1.0 / z if abs(z) > 1e-12 else 0.0
        y_shunt = complex(0, b / 2.0)

        V_from = vmag[from_idx] * np.exp(1j * vang[from_idx])
        V_to = vmag[to_idx] * np.exp(1j * vang[to_idx])

        I_from = (V_from - V_to) * y_series + V_from * y_shunt
        I_to = (V_to - V_from) * y_series + V_to * y_shunt

        S_from = V_from * np.conj(I_from)
        S_to = V_to * np.conj(I_to)

        p0[i] = S_from.real
        q0[i] = S_from.imag
        p1[i] = -S_to.real
        q1[i] = -S_to.imag

    return {"p0": p0, "p1": p1, "q0": q0, "q1": q1}


def calculate_line_flows_from_delta_theta(
    delta_theta: np.ndarray,
    vmag: np.ndarray,
    edge_index_fwd: np.ndarray,
    edge_attr_fwd: np.ndarray,
) -> dict:
    """
    Compute line flows directly from Δθ without θ reconstruction.
    Uses π-model: P_from = V_from²×(g_s+g_sh) - V_from×V_to×(g_s×cos(Δθ) + b_s×sin(Δθ))

    Args:
        delta_theta:    [E_fwd] numpy — Δθ per forward edge (θ_from - θ_to)
        vmag:           [N] numpy — voltage magnitudes
        edge_index_fwd: [2, E_fwd] numpy — forward edge source/target
        edge_attr_fwd:  [E_fwd, 7] numpy — [r, x, b_half, tap, g_ser, b_ser, switch_active]

    Returns:
        dict with keys: 'p0', 'q0', 'p1', 'q1' (from/to in per-unit)
    """
    from_bus = edge_index_fwd[0]
    to_bus = edge_index_fwd[1]
    v_from = vmag[from_bus]
    v_to = vmag[to_bus]

    r = edge_attr_fwd[:, 0]
    x_imp = edge_attr_fwd[:, 1]
    b_half = edge_attr_fwd[:, 2]
    tap = edge_attr_fwd[:, 3]

    z = r + 1j * x_imp
    y_s = 1.0 / z
    g_s = y_s.real
    b_s = y_s.imag

    tap_nz = np.where(np.abs(tap) > 1e-12, tap, 1.0)
    v_from_eff = v_from / tap_nz
    b_sh = b_half

    cos_dt = np.cos(delta_theta)
    sin_dt = np.sin(delta_theta)

    p0 = v_from_eff**2 * g_s - v_from_eff * v_to * (g_s * cos_dt + b_s * sin_dt)
    q0 = -v_from_eff**2 * (b_s + b_sh) - v_from_eff * v_to * (g_s * sin_dt - b_s * cos_dt)
    p1 = -(v_to**2 * g_s - v_to * v_from_eff * (g_s * cos_dt - b_s * sin_dt))
    q1 = -(-v_to**2 * (b_s + b_sh) + v_to * v_from_eff * (g_s * sin_dt + b_s * cos_dt))

    return {"p0": p0, "q0": q0, "p1": p1, "q1": q1}


def build_line_results_from_flows(
    network: pypsa.Network,
    flows: dict,
    snapshot_label,
) -> pd.DataFrame:
    """
    Build a line_results-style DataFrame for one snapshot from calculate_line_flows output.
    """
    idx = pd.Index([snapshot_label], name="snapshot")
    cols = pd.MultiIndex.from_product(
        [network.lines.index, ["P0 pu", "P1 pu", "Q0 pu", "Q1 pu"]]
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for i, line in enumerate(network.lines.index):
        df.loc[snapshot_label, (line, "P0 pu")] = flows["p0"][i]
        df.loc[snapshot_label, (line, "P1 pu")] = flows["p1"][i]
        df.loc[snapshot_label, (line, "Q0 pu")] = flows["q0"][i]
        df.loc[snapshot_label, (line, "Q1 pu")] = flows["q1"][i]
    return df
