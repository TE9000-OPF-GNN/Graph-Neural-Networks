import json

PATH = "GNN_Powerflow_V2.7_Analysis.ipynb"

with open(PATH, "r", encoding="utf-8", newline="\n") as f:
    nb = json.load(f)

OLD_SIG = (
    "    figsize_per_subplot=(10, 7),\n"
    "    max_cols=3,\n"
)
NEW_SIG = (
    "    figsize_per_subplot=(10, 7),\n"
    "    max_cols=None,  # None = no wrapping; int wraps each metric row-block into grid rows of at most max_cols\n"
)

OLD_LAYOUT = (
    "    # --- Grid layout ---  # [STSI 010626]:2-row grid when y_metrics_row2 provided\n"
    "    if y_metrics_row2 is not None:\n"
    "        ncols = max(len(y_metrics), len([m for m in y_metrics_row2 if m is not None]))\n"
    "        nrows = 2\n"
    "    else:\n"
    "        ncols = min(len(y_metrics), max_cols)\n"
    "        nrows = math.ceil(len(y_metrics) / ncols)\n"
    "    fw, fh = figsize_per_subplot\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(fw * ncols, fh * nrows), squeeze=False)\n"
    "\n"
    "    # Build flat list: row/col positions for each metric\n"
    "    if y_metrics_row2 is not None:\n"
    "        _all_metrics = [(ym, 0, ci) for ci, ym in enumerate(y_metrics)]\n"
    "        _all_metrics += [(ym, 1, ci) for ci, ym in enumerate(y_metrics_row2) if ym is not None]\n"
    "    else:\n"
    "        _all_metrics = [(ym, ci // ncols, ci % ncols) for ci, ym in enumerate(y_metrics)]\n"
)
NEW_LAYOUT = (
    "    # --- Grid layout ---  # [STSI 040926]:wrap each row-block by max_cols (mirrors DOE coefficient plot)\n"
    "    def _chunk(seq, size):\n"
    "        seq = [m for m in seq if m is not None]\n"
    "        if not size or size < 1 or size >= len(seq):\n"
    "            return [seq]\n"
    "        return [seq[i:i + size] for i in range(0, len(seq), size)]\n"
    "\n"
    "    # MAE block (y_metrics) and optional RMSE block (y_metrics_row2) chunked independently\n"
    "    _grid_rows = list(_chunk(y_metrics, max_cols))\n"
    "    _row2_clean = [m for m in (y_metrics_row2 or []) if m is not None]\n"
    "    if _row2_clean:\n"
    "        _grid_rows += list(_chunk(_row2_clean, max_cols))\n"
    "\n"
    "    nrows = len(_grid_rows)\n"
    "    ncols = max((len(ch) for ch in _grid_rows), default=1)\n"
    "    fw, fh = figsize_per_subplot\n"
    "    fig, axes = plt.subplots(nrows, ncols, figsize=(fw * ncols, fh * nrows), squeeze=False)\n"
    "\n"
    "    # Build flat list: (metric, row, col) positions\n"
    "    _all_metrics = [(ym, _ri, _ci)\n"
    "                    for _ri, _chunk_row in enumerate(_grid_rows)\n"
    "                    for _ci, ym in enumerate(_chunk_row)]\n"
)

n_sig = n_layout = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    if "def plot_timing_vs_accuracy(" not in src:
        continue
    assert src.count(OLD_SIG) == 1, f"sig match count={src.count(OLD_SIG)}"
    assert src.count(OLD_LAYOUT) == 1, f"layout match count={src.count(OLD_LAYOUT)}"
    src = src.replace(OLD_SIG, NEW_SIG)
    src = src.replace(OLD_LAYOUT, NEW_LAYOUT)
    cell["source"] = src.splitlines(keepends=True)
    n_sig += 1
    n_layout += 1

assert n_sig == 1 and n_layout == 1, f"cell hits sig={n_sig} layout={n_layout}"

with open(PATH, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"OK: patched sig={n_sig} layout={n_layout}")
