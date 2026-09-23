"""Throwaway audit script: dump notebook cell sources containing given markers to text files."""
import json
import os
import sys

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")
OUT_DIR = os.path.join(os.path.dirname(__file__), "_audit_dump")
os.makedirs(OUT_DIR, exist_ok=True)

with open(NOTEBOOK, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

cells = nb["cells"]

markers = [
    ("physicsconfig", "class PhysicsConfig:"),
    ("physics_informed_loss_batch", "def physics_informed_loss_batch("),
    ("compute_physics_residual_edge_local", "def compute_physics_residual_edge_local("),
    ("flow_loss_functions", "def compute_flow_loss_dc_local("),
    ("kvl_and_reconstruct", "def compute_kvl_loop_closure_loss("),
    ("create_graph_data", "def _create_graph_data("),
    ("train_power_flow_gnn", "def train_power_flow_gnn("),
]

for name, marker in markers:
    found = False
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            out_path = os.path.join(OUT_DIR, f"{i:04d}_{name}.py")
            with open(out_path, "w", encoding="utf-8", newline="\n") as out:
                out.write(f"# CELL INDEX {i}\n")
                out.write(src)
            print(f"{name}: cell {i} -> {out_path} ({len(src.splitlines())} lines)")
            found = True
            break
    if not found:
        print(f"{name}: MARKER NOT FOUND: {marker}")
