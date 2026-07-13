"""CSV-based PyPSA network loader.

Source: GNN_Powerflow_V2.6_DataGen.ipynb cell 20
TODO: remove duplicate inline definitions from notebooks once refactored.
"""
# Source: GNN_Powerflow_V2.6_DataGen.ipynb cell 20
# TODO: remove duplicate inline definitions from notebooks once refactored.
from __future__ import annotations

import logging
import os

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def load_system_from_csv(
    base_name: str,
    load_dir: str = "grid_model_files",
) -> pypsa.Network:
    """
    Rebuild a static PyPSA network from CSV tables produced by export_powsybl_to_tables.
    All values in the CSVs are already in p.u. on 100 MVA base — no further scaling needed.

    Parameters
    ----------
    base_name : str  — e.g. "ieee30", "ieee57", "cigre14"
    load_dir  : str  — directory containing the CSV files (default: "grid_model_files")

    Returns
    -------
    pypsa.Network (static, not yet solved)
    """
    base_path = os.path.join(load_dir, base_name)

    buses_df = pd.read_csv(f"{base_path}_buses.csv", index_col=0)
    lines_df = pd.read_csv(f"{base_path}_lines.csv", index_col=0)
    gens_df = pd.read_csv(f"{base_path}_gens.csv", index_col=0)
    loads_df = pd.read_csv(f"{base_path}_loads.csv", index_col=0)

    n = pypsa.Network()
    n.sn_mva = 100.0

    # ── Buses ────────────────────────────────────────────────────────────────
    for bus_idx, row in buses_df.iterrows():
        vmag_pu = float(row.get("v_mag_pu", 1.0) or 1.0)
        n.add(
            "Bus", bus_idx,
            v_nom=1.0,
            v_mag_pu_set=vmag_pu,
            v_mag_pu_min=0.9,
            v_mag_pu_max=1.1,
        )

    # ── Lines (all p.u.) ─────────────────────────────────────────────────────
    for br_id, row in lines_df.iterrows():
        bus0 = row.get("bus1_id")
        bus1 = row.get("bus2_id")
        if not isinstance(bus0, str) or not isinstance(bus1, str):
            continue
        if bus0 not in n.buses.index or bus1 not in n.buses.index:
            continue
        r_val = max(float(row.get("r", 1e-4) or 1e-4), 1e-6)
        x_val = max(float(row.get("x", 1e-4) or 1e-4), 1e-4)
        b_val = max(
            float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0),
            0.0,
        )
        n.add("Line", br_id, bus0=bus0, bus1=bus1, r=r_val, x=x_val, b=b_val / 2, s_nom=1.0)

    # ── Transformers (all p.u.) ───────────────────────────────────────────────
    trafo_path = f"{base_path}_transformers.csv"
    if os.path.exists(trafo_path):
        trafos_df = pd.read_csv(trafo_path, index_col=0)
        for tr_id, row in trafos_df.iterrows():
            bus0 = row.get("bus1_id")
            bus1 = row.get("bus2_id")
            if not isinstance(bus0, str) or not isinstance(bus1, str):
                continue
            if bus0 not in n.buses.index or bus1 not in n.buses.index:
                continue
            x_val = max(float(row.get("x", 0.01) or 0.01), 1e-4)
            r_val = max(float(row.get("r", 0.0) or 0.0), 0.0)
            rated_s = row.get("rated_s", None)
            s_nom = (
                float(rated_s)
                if rated_s is not None and not pd.isna(float(rated_s))
                else 1.0
            )
            n.add(
                "Transformer", tr_id,
                bus0=bus0, bus1=bus1,
                x=x_val, r=r_val, s_nom=s_nom, tap_ratio=1.0,
            )

    # ── Shunt compensators (optional) ────────────────────────────────────────
    shunt_path = f"{base_path}_shunts.csv"
    if os.path.exists(shunt_path):
        shunts_df = pd.read_csv(shunt_path, index_col=0)
        for sh_id, row in shunts_df.iterrows():
            bus_name = row.get("bus_id")
            if not isinstance(bus_name, str) or bus_name not in n.buses.index:
                continue
            b_val = float(row.get("b", 0.0) or 0.0)
            if b_val != 0.0:
                n.add("ShuntImpedance", sh_id, bus=bus_name, b=b_val)

    # ── Generators (all p.u.) ────────────────────────────────────────────────
    slack_assigned = False
    for gen_id, row in gens_df.iterrows():
        bus_name = row.get("bus_id")
        if not isinstance(bus_name, str) or bus_name not in n.buses.index:
            continue
        p_nom = float(row.get("max_p", 1.0) or 1.0)
        p_min = float(row.get("min_p", 0.0) or 0.0)
        p_set = float(row.get("target_p", 0.0) or 0.0)
        q_set = float(row.get("target_q", 0.0) or 0.0)
        regulating = bool(row.get("voltage_regulator_on", False))
        if not slack_assigned:
            ctrl = "Slack"
            slack_assigned = True
        elif regulating:
            ctrl = "PV"
        else:
            ctrl = "PQ"
        n.add(
            "Generator", gen_id,
            bus=bus_name,
            p_nom=p_nom,
            p_min_pu=p_min / max(p_nom, 1e-3),
            p_set=p_set,
            q_set=q_set,
            control=ctrl,
        )

    # Set buses['type'] from generator control for consistency
    for gen_id in n.generators.index:
        bus_name = n.generators.loc[gen_id, "bus"]
        ctrl = n.generators.loc[gen_id, "control"]
        if ctrl in ("Slack", "PV"):
            n.buses.loc[bus_name, "type"] = ctrl
    for bus in n.buses.index:
        if n.buses.loc[bus, "type"] == "":
            n.buses.loc[bus, "type"] = "PQ"

    # ── Loads (all p.u.) ─────────────────────────────────────────────────────
    for load_id, row in loads_df.iterrows():
        bus_name = row.get("bus_id")
        if not isinstance(bus_name, str) or bus_name not in n.buses.index:
            continue
        p0 = float(row.get("p0", row.get("p", 0.0)) or 0.0)
        q0 = float(row.get("q0", row.get("q", 0.0)) or 0.0)
        n.add("Load", load_id, bus=bus_name, p_set=p0, q_set=q0)

    logger.info(
        f"{base_name}: loaded {len(n.buses)} buses, {len(n.generators)} generators, "
        f"{len(n.loads)} loads, {len(n.lines)} lines, {len(n.transformers)} transformers"
    )
    return n
