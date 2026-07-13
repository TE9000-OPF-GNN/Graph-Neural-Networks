"""PTDF and Y-matrix computation for PyPSA networks.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 9
TODO: remove duplicate inline definitions from notebooks once refactored.
"""
# Source: GNN_Powerflow_V2.7_Training.ipynb cell 9
# TODO: remove duplicate inline definitions from notebooks once refactored.
from __future__ import annotations

import copy
import logging

import numpy as np
import pypsa
import torch

logger = logging.getLogger(__name__)


def compute_ptdf_matrix(network: pypsa.Network, ptdf_branch_mode: str = "lines") -> np.ndarray:
    """
    Compute the Power Transfer Distribution Factor (PTDF) matrix for a PyPSA network.
    Returns a numpy array indexed by branches, columns by buses.

    Args:
        ptdf_branch_mode: "lines" returns [n_lines, n_buses],
                          "all"   returns [n_lines + n_trafos, n_buses].
    """
    buses = list(network.buses.index)
    lines = list(network.lines.index)
    n_buses = len(buses)
    n_lines = len(lines)
    n_trafos = len(network.transformers)
    n_branches = n_lines + n_trafos if ptdf_branch_mode == "all" else n_lines

    bus_to_idx = {b: i for i, b in enumerate(buses)}

    B = np.zeros((n_buses, n_buses))
    A = np.zeros((n_branches, n_buses))

    for i, line in enumerate(lines):
        b_val = (1.0 / network.lines.loc[line, "x"]
                 if network.lines.loc[line, "x"] != 0 else 0.0)
        from_idx = bus_to_idx[network.lines.loc[line, "bus0"]]
        to_idx = bus_to_idx[network.lines.loc[line, "bus1"]]
        B[from_idx, from_idx] += b_val
        B[to_idx, to_idx] += b_val
        B[from_idx, to_idx] -= b_val
        B[to_idx, from_idx] -= b_val
        A[i, from_idx] = b_val
        A[i, to_idx] = -b_val

    for trafo_i, trafo in enumerate(network.transformers.index):
        row = network.transformers.loc[trafo]
        if ("x_pu_eff" in row.index and row.get("x_pu_eff") is not None
                and not (x_check := float(row["x_pu_eff"])) == 0):
            x_val = float(row["x_pu_eff"])
        else:
            x_val = float(row["x"])
        if x_val == 0:
            continue
        b_val = 1.0 / x_val
        from_idx = bus_to_idx[row["bus0"]]
        to_idx = bus_to_idx[row["bus1"]]
        B[from_idx, from_idx] += b_val
        B[to_idx, to_idx] += b_val
        B[from_idx, to_idx] -= b_val
        B[to_idx, from_idx] -= b_val
        if ptdf_branch_mode == "all":
            A[n_lines + trafo_i, from_idx] = b_val
            A[n_lines + trafo_i, to_idx] = -b_val

    slack_bus = None
    for gen in network.generators.index:
        if network.generators.loc[gen, "control"] == "Slack":
            slack_bus = network.generators.loc[gen, "bus"]
            break
    if slack_bus is None:
        slack_bus = buses[0]
    slack_idx = bus_to_idx[slack_bus]

    non_slack = [i for i in range(n_buses) if i != slack_idx]
    B_red = B[np.ix_(non_slack, non_slack)]
    A_red = A[:, non_slack]

    try:
        B_red_inv = np.linalg.inv(B_red)
    except np.linalg.LinAlgError:
        B_red_inv = np.linalg.pinv(B_red)

    PTDF_red = A_red @ B_red_inv

    PTDF = np.zeros((n_branches, n_buses))
    non_slack_col = 0
    for j in range(n_buses):
        if j != slack_idx:
            PTDF[:, j] = PTDF_red[:, non_slack_col]
            non_slack_col += 1

    return PTDF


def compute_admittance_matrix(
    network: pypsa.Network,
    device: str = "cpu",
    return_format: str = "torch",
):
    """
    Compute bus admittance matrix Y for a PyPSA network.

    Args:
        return_format: 'torch'         → (Y_real, Y_imag) as torch tensors
                       'numpy'         → complex numpy array
                       'complex_torch' → complex torch tensor
    """
    num_buses = len(network.buses)
    buses = list(network.buses.index)
    bus_to_idx = {b: i for i, b in enumerate(buses)}

    Y = np.zeros((num_buses, num_buses), dtype=complex)

    for line in network.lines.index:
        row = network.lines.loc[line]
        r = row["r"]
        x = row["x"]
        b = row["b"] if "b" in network.lines.columns else 0.0
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y_series = 1.0 / z
        y_shunt = complex(0.0, b / 2.0)
        fi = bus_to_idx[row["bus0"]]
        ti = bus_to_idx[row["bus1"]]
        Y[fi, fi] += y_series
        Y[ti, ti] += y_series
        Y[fi, ti] -= y_series
        Y[ti, fi] -= y_series
        Y[fi, fi] += y_shunt
        Y[ti, ti] += y_shunt

    for trafo in network.transformers.index:
        row = network.transformers.loc[trafo]
        if ("x_pu_eff" in row.index
                and row.get("x_pu_eff") is not None
                and float(row["x_pu_eff"]) != 0.0):
            x = float(row["x_pu_eff"])
            r = float(row["r_pu_eff"]) if "r_pu_eff" in row.index else 0.0
        else:
            x = float(row["x"])
            r = float(row.get("r", 0.0))
        tap = float(row["tap_ratio"]) if "tap_ratio" in row.index else 1.0
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y_series = 1.0 / z
        fi = bus_to_idx[row["bus0"]]
        ti = bus_to_idx[row["bus1"]]
        t = complex(tap, 0.0)
        Y[fi, fi] += y_series / (t * t.conjugate())
        Y[ti, ti] += y_series
        Y[fi, ti] -= y_series / t.conjugate()
        Y[ti, fi] -= y_series / t

    if return_format == "numpy":
        return Y
    elif return_format == "complex_torch":
        return torch.tensor(Y, dtype=torch.complex64, device=device)
    else:
        Y_real = torch.tensor(Y.real, dtype=torch.float32, device=device)
        Y_imag = torch.tensor(Y.imag, dtype=torch.float32, device=device)
        return Y_real, Y_imag


def get_pypsa_Y_numpy(net: pypsa.Network) -> np.ndarray | None:
    """
    Get PyPSA's Ybus via subnetwork API.
    Works on a deepcopy to avoid mutating training network objects.
    """
    if hasattr(net, "admittance_matrix"):
        Y = net.admittance_matrix()
        return np.asarray(Y.todense())

    if not hasattr(net, "determine_network_topology"):
        return None

    try:
        net_copy = copy.deepcopy(net)
        net_copy.determine_network_topology()
    except Exception as e:
        logger.debug(f"[YCHECK] determine_network_topology failed: {e}")
        return None

    if not hasattr(net_copy, "sub_networks") or len(net_copy.sub_networks) == 0:
        return None

    sn_label = list(net_copy.sub_networks.index)[0]
    sn = net_copy.sub_networks.at[sn_label, "obj"]

    if not hasattr(sn, "calculate_Y"):
        return None

    try:
        sn.calculate_Y()
        Y_sparse = sn.Y
        Y_dense = np.asarray(Y_sparse.todense())

        sn_buses_i = list(sn.buses_i())
        sn_buses_o = list(sn.buses_o)
        if sn_buses_i == sn_buses_o:
            return Y_dense
        pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
        perm = [pos_in_o[b] for b in sn_buses_i]
        return Y_dense[np.ix_(perm, perm)]
    except Exception as e:
        logger.debug(f"[YCHECK] calculate_Y() failed: {e}")
        return None


def precompute_Y_matrices(
    networks: list,
    device: str | torch.device = "cpu",
    y_tolerance: float = 1e-6,
    y_matrix_source: str = "auto",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Pre-compute Y matrices for a list of networks.

    y_matrix_source:
      "manual" — always use compute_admittance_matrix.
      "pypsa"  — always use PyPSA's subnetwork Y; raise if not available.
      "auto"   — use PyPSA Y if available and consistent, else manual.
    """
    assert y_matrix_source in ("manual", "pypsa", "auto"), \
        f"y_matrix_source must be 'manual', 'pypsa', or 'auto', got {y_matrix_source!r}"

    if isinstance(device, str):
        device = torch.device(device)

    Y_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    n_manual = 0
    n_pypsa = 0
    n_diff = 0

    for idx, net in enumerate(networks):
        Y_manual = compute_admittance_matrix(net, device="cpu", return_format="numpy")
        Y_pypsa = get_pypsa_Y_numpy(net)

        if y_matrix_source == "manual":
            Y_use = Y_manual
            n_manual += 1
        elif y_matrix_source == "pypsa":
            if Y_pypsa is None:
                raise RuntimeError(
                    f"y_matrix_source='pypsa' but PyPSA Y not available for network {idx}."
                )
            Y_use = Y_pypsa
            n_pypsa += 1
        else:  # "auto"
            if Y_pypsa is None:
                Y_use = Y_manual
                n_manual += 1
            elif Y_manual.shape != Y_pypsa.shape:
                Y_use = Y_pypsa
                n_pypsa += 1
                n_diff += 1
            else:
                max_diff = np.max(np.abs(Y_manual - Y_pypsa))
                if max_diff > y_tolerance:
                    Y_use = Y_pypsa
                    n_pypsa += 1
                    n_diff += 1
                else:
                    Y_use = Y_manual
                    n_manual += 1

        Y_real = torch.from_numpy(Y_use.real).to(device=device, dtype=torch.float32)
        Y_imag = torch.from_numpy(Y_use.imag).to(device=device, dtype=torch.float32)
        Y_list.append((Y_real, Y_imag))

    if n_diff > 0:
        logger.warning(
            f"[YCHECK] Summary ({y_matrix_source}): {n_pypsa} PyPSA Y, "
            f"{n_manual} manual Y, {n_diff} had mismatches > {y_tolerance:.1e}."
        )
    else:
        logger.info(
            f"[YCHECK] Summary ({y_matrix_source}): {n_pypsa} PyPSA Y, "
            f"{n_manual} manual Y, 0 mismatches."
        )

    return Y_list
