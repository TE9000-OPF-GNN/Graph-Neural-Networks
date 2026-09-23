"""Throwaway audit script: print first 3 lines of each code cell 0-16 to check for side effects."""
import json, os

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")
with open(NOTEBOOK, "r", encoding="utf-8") as fh:
    nb = json.load(fh)

for i, cell in enumerate(nb["cells"][:17]):
    if cell.get("cell_type") != "code":
        print(f"--- cell {i}: MARKDOWN ---")
        continue
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    print(f"--- cell {i}: {len(lines)} lines ---")
    for l in lines[:5]:
        print("   ", l)
