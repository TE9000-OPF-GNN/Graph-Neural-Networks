"""Power flow solving and sanity checks for PyPSA networks.

Source: GNN_Powerflow_V2.6_DataGen.ipynb cell 17
TODO: remove duplicate inline definitions from notebooks once refactored.
"""
# Source: GNN_Powerflow_V2.6_DataGen.ipynb cell 17
# TODO: remove duplicate inline definitions from notebooks once refactored.
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pypsa

logger = logging.getLogger(__name__)


def solve_power_flow_with_slack(
    network: pypsa.Network,
    gen_buses: list,
    distribute_slack_flag: bool,
    distributed_slack_mode: str = "proportional",
    custom_slack_weights: dict | None = None,
) -> None:
    """
    Run PyPSA AC power flow, optionally distributing slack across generators.

    Args:
        network:               PyPSA network (modified in-place by pf())
        gen_buses:             list of generator bus IDs (int) for distributed slack
        distribute_slack_flag: if True, use distribute_slack=True in pf()
        distributed_slack_mode: "proportional" | "equal" | "custom"
        custom_slack_weights:  {bus_id: weight} for mode="custom"

    Raises:
        RuntimeError: if network is disconnected before calling pf()
    """
    import networkx as nx

    def _bus_to_int(bus_name: str, fallback: dict) -> int:
        """Parse 'Bus 3' → 3, 'B30' → 30, else assign stable fallback ID."""
        if isinstance(bus_name, str):
            b = bus_name.strip()
            if b.lower().startswith("bus"):
                parts = b.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    return int(parts[-1])
            if b.startswith("B"):
                num = b[1:].strip()
                if num.isdigit():
                    return int(num)
        if bus_name not in fallback:
            fallback[bus_name] = len(fallback) + 1
        return fallback[bus_name]

    G = nx.Graph()
    for _, line in network.lines.iterrows():
        G.add_edge(line.bus0, line.bus1)
    for _, trafo in network.transformers.iterrows():
        G.add_edge(trafo.bus0, trafo.bus1)
    for bus in network.buses.index:
        G.add_node(bus)

    if not nx.is_connected(G):
        raise RuntimeError(
            f"Network disconnected "
            f"({nx.number_connected_components(G)} components) — skip PF"
        )

    if not distribute_slack_flag:
        network.pf(use_seed=True)
        return

    if not gen_buses:
        fallback: dict = {}
        gen_bus_names = sorted(set(network.generators.bus.values))
        gen_buses = [_bus_to_int(b, fallback) for b in gen_bus_names]

    if not gen_buses:
        network.pf(use_seed=True)
        return

    if "slack_weight" not in network.generators.columns:
        network.generators["slack_weight"] = 0.0

    primary_slack_bus = gen_buses[0]
    fallback = {}
    candidates = []
    for g in network.generators.index:
        bus_name = network.generators.loc[g, "bus"]
        bus_id = _bus_to_int(bus_name, fallback)
        if bus_id == primary_slack_bus:
            continue
        candidates.append((g, bus_id))

    if distributed_slack_mode == "proportional":
        total = 0.0
        for g, _ in candidates:
            w = float(network.generators.loc[g, "p_nom"])
            network.generators.loc[g, "slack_weight"] = w
            total += w
        if total > 0:
            network.generators["slack_weight"] /= total
    elif distributed_slack_mode == "equal":
        if candidates:
            w = 1.0 / len(candidates)
            for g, _ in candidates:
                network.generators.loc[g, "slack_weight"] = w
    elif distributed_slack_mode == "custom":
        if custom_slack_weights is None:
            raise ValueError("custom slack requires custom_slack_weights")
        total = 0.0
        for g, bus_id in candidates:
            if bus_id in custom_slack_weights:
                w = float(custom_slack_weights[bus_id])
                network.generators.loc[g, "slack_weight"] = w
                total += w
        if total > 0:
            network.generators["slack_weight"] /= total
    else:
        raise ValueError(f"Unknown distributed_slack_mode: {distributed_slack_mode}")

    network.pf(use_seed=True, distribute_slack=True)


def sanity_check_power_flow(
    network: pypsa.Network,
    base_system: str,
    require_connected: bool = True,
    min_loads: int = 1,
    voltage_max_threshold: float = 1.5,
    voltage_min_threshold: float = 0.5,
    max_angle: float = 180.0,
    max_balance_err: float = 1e-3,
    max_edge_delta_iqr_k: Optional[float] = 5.0,
    max_edge_delta_abs: float = 60.0,
    min_edge_delta_fence: float = 20.0,
    fence_q3_mult: float = 3.0,
) -> None:
    """
    Post-PF sanity checks: connectivity, voltages, angles, power balance, outlier Δθ.

    Raises RuntimeError if any check fails.
    """
    import networkx as nx
    import pandas as pd

    # 1) Connectivity
    G = nx.Graph()
    for _, line in network.lines.iterrows():
        G.add_edge(line["bus0"], line["bus1"])
    for _, trafo in network.transformers.iterrows():
        G.add_edge(trafo["bus0"], trafo["bus1"])

    if require_connected and len(G.nodes()) > 0 and not nx.is_connected(G):
        raise RuntimeError(f"{base_system}: network is disconnected after PF")

    # 2) Voltage sanity
    if hasattr(network, "buses_t") and hasattr(network.buses_t, "v_mag_pu"):
        v = network.buses_t.v_mag_pu.values.flatten()
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"{base_system}: non-finite bus voltages after PF")
        if not np.all((v > voltage_min_threshold) & (v < voltage_max_threshold)):
            raise RuntimeError(
                f"{base_system}: Voltages out of range "
                f"({voltage_min_threshold}-{voltage_max_threshold}) after PF"
            )
    else:
        logger.warning(f"{base_system}: buses_t.v_mag_pu not available for sanity check")

    # 2b) Angle sanity
    if hasattr(network.buses_t, "v_ang"):
        ang_deg = np.degrees(network.buses_t.v_ang.values.flatten())
        if not np.all(np.isfinite(ang_deg)):
            raise RuntimeError(f"{base_system}: non-finite bus angles after PF")
        if np.any(np.abs(ang_deg) > max_angle):
            raise RuntimeError(
                f"{base_system}: angle out of range "
                f"(max |ang|={np.abs(ang_deg).max():.1f}° > {max_angle}°)"
            )

    # 2d) Edge delta-theta outlier detection
    if max_edge_delta_iqr_k is not None and hasattr(network.buses_t, "v_ang"):
        v_ang_rad = network.buses_t.v_ang
        branches = pd.concat(
            [network.lines[["bus0", "bus1"]], network.transformers[["bus0", "bus1"]]],
            ignore_index=True,
        )
        valid = (
            branches["bus0"].isin(v_ang_rad.columns)
            & branches["bus1"].isin(v_ang_rad.columns)
        )
        branches = branches[valid]
        if len(branches) >= 3:
            ang0 = v_ang_rad[branches["bus0"].values].values
            ang1 = v_ang_rad[branches["bus1"].values].values
            delta_deg = np.abs(np.degrees(ang0 - ang1))
            max_per_branch = delta_deg.max(axis=0)
            max_dt = float(max_per_branch.max())
            worst_idx = int(max_per_branch.argmax())
            worst_branch = (
                f"{branches.iloc[worst_idx]['bus0']}→"
                f"{branches.iloc[worst_idx]['bus1']}"
            )
            if max_dt > max_edge_delta_abs:
                raise RuntimeError(
                    f"{base_system}: edge |Δθ|={max_dt:.1f}° > {max_edge_delta_abs}° "
                    f"hard cap on branch {worst_branch}"
                )
            q1, q3 = np.percentile(delta_deg, [25, 75])
            iqr = q3 - q1
            upper_fence = max(
                q3 + max_edge_delta_iqr_k * iqr,
                min_edge_delta_fence,
                q3 * fence_q3_mult,
            )
            if iqr > 0.5 and max_dt > upper_fence:
                raise RuntimeError(
                    f"{base_system}: edge |Δθ| outlier: max={max_dt:.1f}° > "
                    f"fence={upper_fence:.1f}° (Q3={q3:.1f}°, IQR={iqr:.1f}°, "
                    f"k={max_edge_delta_iqr_k}) on branch {worst_branch}"
                )

    # 3) Slack generator presence
    slack_gens = network.generators.index[network.generators["control"] == "Slack"]
    if len(slack_gens) == 0:
        logger.warning(f"{base_system}: no explicit Slack generator found")

    # 4) Minimum loads
    if len(network.loads) < min_loads:
        raise RuntimeError(
            f"{base_system}: insufficient loads ({len(network.loads)}) after PF"
        )

    # 5) System power balance (KCL)
    if hasattr(network, "generators_t") and hasattr(network.generators_t, "p"):
        gen_p = float(network.generators_t.p.values.sum())
        load_p = float(network.loads_t.p.values.sum())
        line_loss = float((network.lines_t.p0.values + network.lines_t.p1.values).sum())
        traf_arr = network.transformers_t.p0.values + network.transformers_t.p1.values
        traf_loss = float(traf_arr.sum()) if traf_arr.size > 0 else 0.0
        balance_err = abs(gen_p - load_p - line_loss - traf_loss) / max(abs(gen_p), 1e-6)
        if balance_err > max_balance_err:
            raise RuntimeError(
                f"{base_system}: power balance error {balance_err:.2e} > "
                f"{max_balance_err:.0e} "
                f"(gen={gen_p:.4f}, load={load_p:.4f}, losses={line_loss + traf_loss:.6f})"
            )

    logger.info(
        f"{base_system}: PF sanity OK "
        f"(buses={len(network.buses)}, gens={len(network.generators)}, "
        f"loads={len(network.loads)})"
    )
