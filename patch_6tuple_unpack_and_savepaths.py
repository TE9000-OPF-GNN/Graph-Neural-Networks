"""
Patch GNN_Powerflow_V2.6.1_Training.ipynb:
  1. Fix val loop and test loop 4-tuple unpacks → 6-tuple  (Cell 15)
  2. Refresh S1-S9 sweep save paths  _200526.json → _210526.json  (Cells 27-35)
"""
import json, re

NB = "GNN_Powerflow_V2.6.1_Training.ipynb"

OLD_UNPACK = "_, _, physics, angle_ref = physics_informed_loss_batch("
NEW_UNPACK = "_, _, physics, angle_ref, _, _ = physics_informed_loss_batch(  # [STSI 210526]: unpack 6-tuple"

OLD_DATE = "_200526.json"
NEW_DATE = "_210526.json"

with open(NB, encoding="utf-8", newline="\n") as f:
    raw = f.read()

nb = json.loads(raw)

unpack_fixed = 0
date_fixed = 0

for cell in nb["cells"]:
    src_lines = cell.get("source", [])
    new_lines = []
    for line in src_lines:
        changed = False
        if OLD_UNPACK in line:
            line = line.replace(OLD_UNPACK, NEW_UNPACK)
            unpack_fixed += 1
            changed = True
        if OLD_DATE in line:
            line = line.replace(OLD_DATE, NEW_DATE)
            date_fixed += 1
            changed = True
        new_lines.append(line)
    cell["source"] = new_lines

print(f"unpack fixes applied : {unpack_fixed}  (expected 2)")
print(f"date path fixes applied: {date_fixed}  (expected 9)")

if unpack_fixed != 2 or date_fixed != 9:
    print("ERROR: unexpected counts — aborting write")
    raise SystemExit(1)

out = json.dumps(nb, ensure_ascii=False, indent=1)
with open(NB, "w", encoding="utf-8", newline="\n") as f:
    f.write(out)

print("Patch written successfully.")
