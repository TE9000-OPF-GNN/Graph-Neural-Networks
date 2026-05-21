"""
Patch 013: Fix get_pypsa_Y_numpy buses_o->buses_i bus ordering.

Replaces the broken type-mismatched guard (always returned buses_o order)
with correct permutation logic that maps buses_o -> buses_i.

Targets: GNN_Powerflow_V2.6.1_Training.ipynb
         GNN_Powerflow_V2.6_Training.ipynb
"""

import re

OLD_BLOCK = (
    '        sn_bus_order  = list(sn.buses_i())\\n",\n'
    '    "        net_bus_order = list(range(len(net_copy.buses)))\\n",\n'
    '    "\\n",\n'
    '    "        if sorted(sn_bus_order) == net_bus_order:\\n",\n'
    '    "            return Y_dense[np.ix_(sn_bus_order, sn_bus_order)]\\n",\n'
    '    "        else:\\n",\n'
    '    "            return Y_dense\\n",'
)

NEW_BLOCK = (
    '        # [STSI 210526]: fixed type-mismatch guard; was always returning Y in buses_o order'
    ' (str vs int comparison always False) -- reorder buses_o->buses_i instead\\n",\n'
    '    "        sn_buses_i = list(sn.buses_i())\\n",\n'
    '    "        sn_buses_o = list(sn.buses_o)\\n",\n'
    '    "        if sn_buses_i == sn_buses_o:\\n",\n'
    '    "            return Y_dense\\n",\n'
    '    "        pos_in_o = {b: j for j, b in enumerate(sn_buses_o)}\\n",\n'
    '    "        perm = [pos_in_o[b] for b in sn_buses_i]\\n",\n'
    '    "        return Y_dense[np.ix_(perm, perm)]\\n",'
)

TARGETS = [
    "GNN_Powerflow_V2.6.1_Training.ipynb",
    "GNN_Powerflow_V2.6_Training.ipynb",
]

for fname in TARGETS:
    with open(fname, encoding="utf-8", newline="\n") as f:
        content = f.read()

    count = content.count(OLD_BLOCK)
    if count == 0:
        print(f"ERROR: OLD_BLOCK not found in {fname} — check pattern")
        continue
    if count > 1:
        print(f"WARNING: OLD_BLOCK found {count} times in {fname} — expected 1")

    new_content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)

    with open(fname, encoding="utf-8", newline="\n") as f:
        pass  # just checking it opens

    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)

    print(f"Patched {fname} ({count} replacement(s))")

# Verify: old pattern should be gone from both files
for fname in TARGETS:
    with open(fname, encoding="utf-8", newline="\n") as f:
        content = f.read()
    old_count = content.count("sn_bus_order  = list(sn.buses_i())")
    new_count = content.count("sn_buses_i = list(sn.buses_i())")
    print(f"  {fname}: old_pattern={old_count} (expect 0), new_pattern={new_count} (expect 1)")
