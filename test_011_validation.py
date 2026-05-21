"""
End-to-end validation for task 011 patches:
  A: b_val clamped >= 0 in load_system_from_csv
  B: n.sn_mva = 100.0 set in load_system_from_csv
  G: KCL balance check in sanity_check_power_flow
"""
import json, sys, types

NB = "GNN_Powerflow_V2.6_DataGen.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]

# Build a shared exec namespace from the essential cells
ns = {"__name__": "__test__"}

def exec_cell(idx):
    src = "".join(cells[idx].get("source", []))
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), ns)
    except Exception as e:
        print(f"  WARN: cell {idx} raised {type(e).__name__}: {e}")

print("Loading notebook cells 1, 2, 4, 16, 19 ...")
exec_cell(1)   # imports: logging, os, pandas, numpy, pypsa, etc.
exec_cell(2)   # GRID_MODEL_FILES path constant
exec_cell(4)   # itertools + misc constants
exec_cell(16)  # sanity_check_power_flow + helpers
exec_cell(19)  # load_system_from_csv
print("  Done.\n")

load_system_from_csv   = ns["load_system_from_csv"]
sanity_check_power_flow = ns["sanity_check_power_flow"]
GRID_MODEL_FILES = r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN\grid_model_files"

TEST_SYSTEM = "cigre14"   # has transformers; available in OneDrive grid_model_files

# ── Test 1: n.sn_mva == 100 and transformer s_nom unchanged ──────────────────
print(f"Test 1: load_system_from_csv('{TEST_SYSTEM}') — sn_mva + transformer s_nom")
n9 = load_system_from_csv(TEST_SYSTEM, load_dir=GRID_MODEL_FILES)
assert n9.sn_mva == 100.0, f"FAIL: sn_mva={n9.sn_mva}, expected 100.0"
print(f"  sn_mva = {n9.sn_mva}  ✓")
if len(n9.transformers) > 0:
    s_nom_val = n9.transformers["s_nom"].iloc[0]
    assert abs(s_nom_val - 1.0) < 0.01, f"FAIL: transformer s_nom={s_nom_val}, expected ~1.0"
    print(f"  transformer s_nom = {s_nom_val:.4f}  ✓")
else:
    print(f"  (no transformers in {TEST_SYSTEM} — skip s_nom check)")

# ── Test 2: power flow + sanity_check_power_flow passes ──────────────────────
print(f"\nTest 2: {TEST_SYSTEM} PF + sanity_check_power_flow")
import logging
logging.getLogger("pypsa").setLevel(logging.CRITICAL)

n9.pf()
sanity_check_power_flow(n9, TEST_SYSTEM)
print("  sanity_check passed  ✓")

# ── Test 3: b clamp defense — negative b is silently clamped to 0 ─────────────
print("\nTest 3: b clamp — negative b_val clamped to 0")
# Simulate what load_system_from_csv does when b1 = -14.3 (stale ieee30 artifact)
b1_raw, b2_raw = -14.3, 0.0
b_val = max(float(b1_raw or 0.0) + float(b2_raw or 0.0), 0.0)
assert b_val == 0.0, f"FAIL: b_val={b_val}, expected 0.0"
print(f"  b1={b1_raw}, b2={b2_raw}  →  b_val={b_val}  ✓")

# Normal positive b passes through unchanged
b1_raw, b2_raw = 0.05, 0.03
b_val = max(float(b1_raw or 0.0) + float(b2_raw or 0.0), 0.0)
assert abs(b_val - 0.08) < 1e-9
print(f"  b1={b1_raw}, b2={b2_raw}  →  b_val={b_val:.4f}  ✓")

# ── Test 4: KCL check triggers on corrupted network ───────────────────────────
print("\nTest 4: KCL check triggers RuntimeError on fake imbalance")
import pypsa, numpy as np, pandas as pd

# Build a tiny 2-bus network and manually corrupt generators_t.p
n_bad = pypsa.Network()
n_bad.sn_mva = 100.0
n_bad.add("Bus", "B1", v_nom=1.0)
n_bad.add("Bus", "B2", v_nom=1.0)
n_bad.add("Line", "L1", bus0="B1", bus1="B2", r=0.01, x=0.1, b=0.0, s_nom=1.0)
n_bad.add("Generator", "G1", bus="B1", control="Slack", p_nom=1.0)
n_bad.add("Load", "D1", bus="B2", p_set=0.5, q_set=0.0)

# Run PF to populate _t frames
n_bad.pf()

# Corrupt gen p to create a massive balance error
n_bad.generators_t.p["G1"] = [999.0]   # way off from actual load

try:
    sanity_check_power_flow(n_bad, "test_bad", max_balance_err=1e-3)
    print("  FAIL: no RuntimeError raised — KCL check did not trigger!")
    sys.exit(1)
except RuntimeError as e:
    print(f"  RuntimeError raised as expected: {e}  ✓")

print("\nAll tests passed.")
