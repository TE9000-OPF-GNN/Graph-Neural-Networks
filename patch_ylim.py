"""Patch: add robust y-axis cap to histogram functions in DataGen notebook."""
import json

NOTEBOOK = "GNN_Powerflow_V2.6_DataGen.ipynb"

# The cap block to inject before tight_layout in each histogram function.
# Uses n_cols defined in the function scope.
CAP_BLOCK = (
    "\n    # Cap shared y-axis per column at 95th percentile of row-maxes"
    "  # [STSI 210526]:prevent single narrow spike from dominating shared y scale\n"
    "    for _ci in range(n_cols):\n"
    "        _ymaxes = [axes[_r, _ci].get_ylim()[1] for _r in range(n_systems)]\n"
    "        _valid = [y for y in _ymaxes if y > 0]\n"
    "        if _valid:\n"
    "            axes[0, _ci].set_ylim(0, np.percentile(_valid, 95))\n"
)

TIGHT = "    fig.tight_layout(rect=[0.08, 0, 1, 0.96])\n    return fig"

nb = json.load(open(NOTEBOOK, encoding="utf-8"))

patched = 0
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "def plot_electrical_params" not in src:
        continue

    # We want to insert CAP_BLOCK before each tight_layout EXCEPT plot_topology_stats
    # Strategy: split on function boundaries, patch each histogram function separately
    funcs = [
        ("def plot_electrical_params(", "def plot_loadgen_patterns("),
        ("def plot_loadgen_patterns(", "def plot_solved_state_matrix("),
        ("def plot_solved_state_matrix(", None),
    ]

    result = src
    for func_start, func_end in funcs:
        # Extract the slice for this function
        i = result.index(func_start)
        j = result.index(func_end, i) if func_end and func_end in result[i:] else len(result)
        func_body = result[i:j]

        # Find the tight_layout line in this slice
        old = "    fig.tight_layout(rect=[0.08, 0, 1, 0.96])\n    return fig"
        if old not in func_body:
            print(f"WARNING: tight_layout not found in {func_start}")
            continue

        new = CAP_BLOCK + old
        func_body_patched = func_body.replace(old, new, 1)
        result = result[:i] + func_body_patched + result[j:]
        patched += 1
        print(f"Patched: {func_start[:40]}")

    cell["source"] = [result]
    break

if patched == 3:
    with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved — {patched} functions patched.")
else:
    print(f"ERROR: expected 3 patches, got {patched}. File NOT saved.")
