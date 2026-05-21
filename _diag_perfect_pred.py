"""
Diagnostic: physics residual with PERFECT predictions.

Tests whether compute_power_flow_residual_from_pred gives ~0 residual
when fed the ground-truth PyPSA solution as 'predictions'.

Compares three Y-matrix sources:
  1. manual    -- compute_admittance_matrix  (buses_i order, no shunts)
  2. pypsa_buggy -- get_pypsa_Y_numpy as-is  (buses_o order = wrong for ieee30/cigre14)
  3. pypsa_fixed -- get_pypsa_Y_numpy with corrected buses_o->buses_i reordering

Run from repo root with: .venv\Scripts\python.exe _diag_perfect_pred.py
"""
import sys, os, copy, json
import numpy as np
import torch
import pypsa

DATA_ROOT = r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN"
GRID_DIR  = r"grid_model_files"

# ── Minimal function copies ────────────────────────────────────────────────────

import pandas as pd

def compute_admittance_matrix_manual(network):
    """Manual Y in network.buses.index order (no shunts)."""
    num_buses = len(network.buses)
    buses = list(network.buses.index)
    bus_to_idx = {b: i for i, b in enumerate(buses)}
    Y = np.zeros((num_buses, num_buses), dtype=complex)

    for line in network.lines.index:
        row = network.lines.loc[line]
        r, x = float(row["r"]), float(row["x"])
        b = float(row["b"]) if "b" in network.lines.columns else 0.0
        z = complex(r, x)
        if abs(z) < 1e-12: continue
        y_s = 1.0 / z
        y_sh = complex(0, b / 2.0)
        fi, ti = bus_to_idx[row["bus0"]], bus_to_idx[row["bus1"]]
        Y[fi,fi] += y_s + y_sh;  Y[ti,ti] += y_s + y_sh
        Y[fi,ti] -= y_s;         Y[ti,fi] -= y_s

    for trafo in network.transformers.index:
        row = network.transformers.loc[trafo]
        if "x_pu_eff" in row.index and pd.notna(row["x_pu_eff"]) and float(row["x_pu_eff"]) != 0.0:
            x = float(row["x_pu_eff"])
            r = float(row.get("r_pu_eff", 0.0))
        else:
            x, r = float(row["x"]), float(row.get("r", 0.0))
        tap = float(row["tap_ratio"]) if "tap_ratio" in row.index else 1.0
        z = complex(r, x)
        if abs(z) < 1e-12: continue
        y_s = 1.0 / z
        t = complex(tap, 0)
        fi, ti = bus_to_idx[row["bus0"]], bus_to_idx[row["bus1"]]
        Y[fi,fi] += y_s / (t * t.conjugate())
        Y[ti,ti] += y_s
        Y[fi,ti] -= y_s / t.conjugate()
        Y[ti,fi] -= y_s / t

    return Y


def get_pypsa_Y_buggy(net):
    """Current (broken) implementation -- returns Y in buses_O order."""
    net_copy = copy.deepcopy(net)
    net_copy.determine_network_topology()
    if not hasattr(net_copy, "sub_networks") or len(net_copy.sub_networks) == 0:
        return None
    sn_label = list(net_copy.sub_networks.index)[0]
    sn = net_copy.sub_networks.at[sn_label, "obj"]
    sn.calculate_Y()
    return np.asarray(sn.Y.todense())   # buses_o order -- BUG


def get_pypsa_Y_fixed(net):
    """Fixed implementation -- reorders to buses_i (network.buses.index) order."""
    net_copy = copy.deepcopy(net)
    net_copy.determine_network_topology()
    if not hasattr(net_copy, "sub_networks") or len(net_copy.sub_networks) == 0:
        return None
    sn_label = list(net_copy.sub_networks.index)[0]
    sn = net_copy.sub_networks.at[sn_label, "obj"]
    sn.calculate_Y()
    Y_dense = np.asarray(sn.Y.todense())

    sn_buses_i = list(sn.buses_i())
    sn_buses_o = list(sn.buses_o)
    if sn_buses_i == sn_buses_o:
        return Y_dense  # already correct
    pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}
    perm = [pos_in_o[b] for b in sn_buses_i]
    return Y_dense[np.ix_(perm, perm)]


def kcl_residual_mse(Y_complex, net, snapshot_idx=0):
    """
    Compute KCL residual MSE using PyPSA's stored solution.
    Returns MSE of (S_calc - S_inj) per bus.
    """
    sn = net.pf()  # run power flow to get solution
    buses = list(net.buses.index)
    n = len(buses)
    bidx = {b: i for i, b in enumerate(buses)}

    # Ground truth voltages from PyPSA solution
    v_mag = np.array([net.buses_t.v_mag_pu[b].iloc[snapshot_idx]
                      if b in net.buses_t.v_mag_pu.columns else 1.0
                      for b in buses])
    v_ang = np.array([net.buses_t.v_ang[b].iloc[snapshot_idx]
                      if b in net.buses_t.v_ang.columns else 0.0
                      for b in buses])   # radians

    v_real = v_mag * np.cos(v_ang)
    v_imag = v_mag * np.sin(v_ang)

    I_real = Y_complex.real @ v_real - Y_complex.imag @ v_imag
    I_imag = Y_complex.real @ v_imag + Y_complex.imag @ v_real

    p_calc = v_real * I_real + v_imag * I_imag
    q_calc = v_imag * I_real - v_real * I_imag

    # Ground truth injections: P_gen - P_load per bus (= buses_t.p)
    p_inj = np.zeros(n)
    q_inj = np.zeros(n)
    if not net.buses_t.p.empty:
        for b in net.buses_t.p.columns:
            if b in bidx:
                p_inj[bidx[b]] = net.buses_t.p[b].iloc[snapshot_idx]
    if not net.buses_t.q.empty:
        for b in net.buses_t.q.columns:
            if b in bidx:
                q_inj[bidx[b]] = net.buses_t.q[b].iloc[snapshot_idx]

    p_res = (p_calc - p_inj) ** 2
    q_res = (q_calc - q_inj) ** 2
    mse = np.mean(p_res + q_res)

    # Report worst buses
    per_bus = p_res + q_res
    worst = np.argsort(per_bus)[-3:][::-1]
    return mse, {buses[i]: float(per_bus[i]) for i in worst}


# ── Load a few networks ────────────────────────────────────────────────────────

TRAIN_SAVED = os.path.join(DATA_ROOT, "training_networks_saved")

def load_networks(dataset_name, n=3):
    """Load first n networks from a saved dataset (JSON manifest + .nc files)."""
    json_path = os.path.join(TRAIN_SAVED, f"{dataset_name}.json")
    if not os.path.exists(json_path):
        print(f"  NOT FOUND: {json_path}")
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    file_list = data.get("files", [])[:n]
    networks = []
    for rel_path in file_list:
        full_path = os.path.join(TRAIN_SAVED, rel_path)
        if not os.path.exists(full_path):
            print(f"  Missing nc file: {full_path}")
            continue
        net = pypsa.Network()
        net.import_from_netcdf(full_path)
        networks.append(net)
    return networks


def test_network(net, label):
    buses_i = list(net.buses.index)

    # Run PF to get ground truth
    try:
        net.pf()
    except Exception as e:
        print(f"  {label}: PF failed — {e}")
        return

    # Check buses_o vs buses_i ordering
    net_copy = copy.deepcopy(net)
    net_copy.determine_network_topology()
    sn_label = list(net_copy.sub_networks.index)[0]
    sn = net_copy.sub_networks.at[sn_label, "obj"]
    buses_o = list(sn.buses_o)
    order_same = (buses_i == buses_o)

    # Build Y matrices
    Y_manual = compute_admittance_matrix_manual(net)
    Y_buggy  = get_pypsa_Y_buggy(net)
    Y_fixed  = get_pypsa_Y_fixed(net)

    # Compute KCL residuals
    def kcl_mse(Y):
        v_mag = np.array([net.buses_t.v_mag_pu.get(b, pd.Series([1.0]))[0] for b in buses_i])
        v_ang = np.array([net.buses_t.v_ang.get(b, pd.Series([0.0]))[0] for b in buses_i])
        vr = v_mag * np.cos(v_ang); vi = v_mag * np.sin(v_ang)
        Ir = Y.real @ vr - Y.imag @ vi
        Ii = Y.real @ vi + Y.imag @ vr
        pc = vr*Ir + vi*Ii; qc = vi*Ir - vr*Ii
        # injections from buses_t
        p_inj = np.array([net.buses_t.p.get(b, pd.Series([0.0]))[0]
                          if b in getattr(net.buses_t.p, 'columns', []) else 0.0
                          for b in buses_i])
        q_inj = np.array([net.buses_t.q.get(b, pd.Series([0.0]))[0]
                          if b in getattr(net.buses_t.q, 'columns', []) else 0.0
                          for b in buses_i])
        return float(np.mean((pc-p_inj)**2 + (qc-q_inj)**2))

    mse_manual = kcl_mse(Y_manual)
    mse_buggy  = kcl_mse(Y_buggy)  if Y_buggy is not None  else float('nan')
    mse_fixed  = kcl_mse(Y_fixed)  if Y_fixed is not None  else float('nan')

    print(f"  {label:45s}  buses_i==buses_o={order_same!s:5}  "
          f"MSE_manual={mse_manual:.3e}  MSE_buggy={mse_buggy:.3e}  MSE_fixed={mse_fixed:.3e}")


# ── Run tests ─────────────────────────────────────────────────────────────────
print("=" * 100)
print("Test: KCL residual with GROUND-TRUTH voltages from PyPSA solution")
print("  MSE ≈ 0   → Y matrix correct for this network type")
print("  MSE >> 0  → Y matrix wrong (either wrong order or missing elements)")
print("=" * 100)

# Load saved networks from different types
test_cases = [
    ("ieee9_large_singel_slack",   "ieee9 (single-slack, 9 buses)"),
    ("cigre14_large",              "cigre14 (~14 buses)"),
    ("ieee30_large",               "ieee30 (~32 buses)"),
]

for dataset_name, label in test_cases:
    print(f"\n--- {label} ---")
    nets = load_networks(dataset_name, n=3)
    if not nets:
        continue
    for i, net in enumerate(nets):
        n_buses = len(net.buses)
        bus_types = net.generators.groupby("control").size().to_dict()
        type_str = f"{n_buses}B({bus_types})"
        test_network(net, f"network[{i}] {type_str}")

print("\nDone.")
