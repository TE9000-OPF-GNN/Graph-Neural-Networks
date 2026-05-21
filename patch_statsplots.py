"""Patch: add plot_loadgen_patterns_stats and plot_solved_state_matrix_stats to DataGen notebook."""
import json

NOTEBOOK = "GNN_Powerflow_V2.6_DataGen.ipynb"

LOADGEN_STATS = '''

def plot_loadgen_patterns_stats(
    networks_dict,
    figsize=(16, 5),
    suptitle: str = "Load & Generation Patterns \\u2014 Distribution by System",
    clip_percentile: tuple = (1, 99),
):  # [STSI 210526]:box-plot companion to plot_loadgen_patterns; one box per system per parameter
    """
    Box plots comparing load/generation distributions across systems.

    One subplot per parameter (Load P, Load Q, Gen P, Gen p_nom, Gen:Load ratio, Load Q/P).
    Each subplot shows one box per system.

    Parameters
    ----------
    networks_dict : dict[str, list[pypsa.Network]]
    figsize : tuple
    suptitle : str
    clip_percentile : tuple[float, float]
        Percentile range for outlier clipping before boxing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    systems = [s for s, nets in networks_dict.items() if len(nets) > 0]
    col_titles = ["Load P [p.u.]", "Load Q [p.u.]", "Gen P_set [p.u.]",
                  "Gen p_nom [p.u.]", "Gen:Load P Ratio", "Load Q/P Ratio"]
    n_cols = len(col_titles)

    # Collect arrays per system per parameter
    all_data = {sys: [] for sys in systems}
    for sys_name in systems:
        nets = networks_dict[sys_name]
        lp_all, lq_all, gp_all, gnom_all, ratio_all, qp_all = [], [], [], [], [], []
        for net in nets:
            n_snaps = len(net.loads_t.p_set) if not net.loads_t.p_set.empty else 1
            if not net.loads_t.p_set.empty:
                lp_all.append(net.loads_t.p_set.values.flatten())
            elif len(net.loads) > 0:
                lp_all.append(net.loads["p_set"].values)
            if not net.loads_t.q_set.empty:
                lq_all.append(net.loads_t.q_set.values.flatten())
            elif len(net.loads) > 0:
                lq_all.append(net.loads["q_set"].values)
            if not net.generators_t.p_set.empty:
                gp_all.append(net.generators_t.p_set.values.flatten())
            elif len(net.generators) > 0:
                gp_all.append(net.generators["p_set"].values)
            if len(net.generators) > 0:
                gnom_all.append(net.generators["p_nom"].values)
            total_gen_p = (net.generators_t.p_set.values.sum(axis=1)
                          if not net.generators_t.p_set.empty
                          else np.full(n_snaps, net.generators["p_set"].sum()))
            total_load_p = (net.loads_t.p_set.values.sum(axis=1)
                           if not net.loads_t.p_set.empty
                           else np.full(n_snaps, net.loads["p_set"].sum()))
            valid_load = np.abs(total_load_p) > 1e-6
            if valid_load.any():
                ratio_all.append(total_gen_p[valid_load] / total_load_p[valid_load])
            if not net.loads_t.p_set.empty and not net.loads_t.q_set.empty:
                lp_v = net.loads_t.p_set.values.flatten()
                lq_v = net.loads_t.q_set.values.flatten()
            elif len(net.loads) > 0:
                lp_v = net.loads["p_set"].values
                lq_v = net.loads["q_set"].values
            else:
                lp_v, lq_v = np.array([]), np.array([])
            vp = np.abs(lp_v) > 1e-6
            if vp.any():
                qp_all.append(lq_v[vp] / lp_v[vp])
        all_data[sys_name] = [
            np.concatenate(lp_all)   if lp_all   else np.array([]),
            np.concatenate(lq_all)   if lq_all   else np.array([]),
            np.concatenate(gp_all)   if gp_all   else np.array([]),
            np.concatenate(gnom_all) if gnom_all else np.array([]),
            np.concatenate(ratio_all) if ratio_all else np.array([]),
            np.concatenate(qp_all)   if qp_all   else np.array([]),
        ]

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col_idx, title in enumerate(col_titles):
        ax = axes[0, col_idx]
        box_data, labels, patch_colors = [], [], []
        for i, sys_name in enumerate(systems):
            arr = all_data[sys_name][col_idx]
            if len(arr) > 0:
                rng = _clip_range(arr, clip_percentile)
                clipped = arr[(arr >= rng[0]) & (arr <= rng[1])] if rng else arr
                box_data.append(clipped)
                labels.append(sys_name)
                patch_colors.append(colors[i % len(colors)])
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5),
                            whiskerprops=dict(lw=1), capprops=dict(lw=1),
                            flierprops=dict(marker=".", ms=2, alpha=0.3))
            for patch, color in zip(bp["boxes"], patch_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return fig


def plot_solved_state_matrix_stats(
    networks_dict,
    figsize=(14, 5),
    suptitle: str = "Solved State Variables \\u2014 Distribution by System",
    clip_percentile: tuple = (1, 99),
):  # [STSI 210526]:box-plot companion to plot_solved_state_matrix; one box per system per variable
    """
    Box plots comparing solved-state distributions across systems.

    One subplot per variable (V_mag, V_ang, P_inject, Q_inject, Line P_flow).
    Each subplot shows one box per system.

    Parameters
    ----------
    networks_dict : dict[str, list[pypsa.Network]]
    figsize : tuple
    suptitle : str
    clip_percentile : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    systems = [s for s, nets in networks_dict.items() if len(nets) > 0]
    col_titles = ["V_mag [p.u.]", "V_ang [\\u00b0]", "P_inject [p.u.]",
                  "Q_inject [p.u.]", "Line P_flow [p.u.]"]
    n_cols = len(col_titles)

    all_data = {sys: [] for sys in systems}
    for sys_name in systems:
        nets = networks_dict[sys_name]
        vm_all, va_all, p_all, q_all, pf_all = [], [], [], [], []
        for net in nets:
            if not net.buses_t.v_mag_pu.empty:
                vm_all.append(net.buses_t.v_mag_pu.values.flatten())
            if not net.buses_t.v_ang.empty:
                va_all.append(net.buses_t.v_ang.values.flatten())
            if not net.buses_t.p.empty:
                p_all.append(net.buses_t.p.values.flatten())
            if not net.buses_t.q.empty:
                q_all.append(net.buses_t.q.values.flatten())
            if not net.lines_t.p0.empty:
                pf_all.append(net.lines_t.p0.values.flatten())
        all_data[sys_name] = [
            np.concatenate(vm_all) if vm_all else np.array([]),
            np.concatenate(va_all) if va_all else np.array([]),
            np.concatenate(p_all)  if p_all  else np.array([]),
            np.concatenate(q_all)  if q_all  else np.array([]),
            np.concatenate(pf_all) if pf_all else np.array([]),
        ]

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col_idx, title in enumerate(col_titles):
        ax = axes[0, col_idx]
        box_data, labels, patch_colors = [], [], []
        for i, sys_name in enumerate(systems):
            arr = all_data[sys_name][col_idx]
            if len(arr) > 0:
                rng = _clip_range(arr, clip_percentile)
                clipped = arr[(arr >= rng[0]) & (arr <= rng[1])] if rng else arr
                box_data.append(clipped)
                labels.append(sys_name)
                patch_colors.append(colors[i % len(colors)])
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                            medianprops=dict(color="black", lw=1.5),
                            whiskerprops=dict(lw=1), capprops=dict(lw=1),
                            flierprops=dict(marker=".", ms=2, alpha=0.3))
            for patch, color in zip(bp["boxes"], patch_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return fig
'''

CALLSITE_OLD = (
    "#fig3 = plot_loadgen_patterns(all_networks,n_bins=n_bins, clip_percentile=(1, 95))\n"
    "#plt.show()\n"
    "#fig4 = plot_solved_state_matrix(all_networks,n_bins=n_bins, clip_percentile=(1, 95))\n"
    "#plt.show()"
)
CALLSITE_NEW = (
    "#fig3 = plot_loadgen_patterns(all_networks,n_bins=n_bins, clip_percentile=(1, 95))\n"
    "#plt.show()\n"
    "fig3b = plot_loadgen_patterns_stats(all_networks)  "
    "# [STSI 210526]:box-plot stats companion for load/gen patterns\n"
    "plt.show()\n"
    "#fig4 = plot_solved_state_matrix(all_networks,n_bins=n_bins, clip_percentile=(1, 95))\n"
    "#plt.show()\n"
    "fig4b = plot_solved_state_matrix_stats(all_networks)  "
    "# [STSI 210526]:box-plot stats companion for solved state\n"
    "plt.show()"
)

nb = json.load(open(NOTEBOOK, encoding="utf-8"))

func_patched = callsite_patched = False

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # 1. Add new functions after plot_solved_state_matrix
    if "def plot_solved_state_matrix" in src and "def plot_loadgen_patterns_stats" not in src:
        cell["source"] = [src + LOADGEN_STATS]
        func_patched = True
        print("Added new stat functions to plotting cell")

    # 2. Update call-site
    if "plot_topology_stats" in src and "fig2b" in src and "fig3b" not in src:
        if CALLSITE_OLD in src:
            cell["source"] = [src.replace(CALLSITE_OLD, CALLSITE_NEW, 1)]
            callsite_patched = True
            print("Updated call-site cell")
        else:
            print("WARNING: call-site OLD string not found")
            # Debug: show the lines with fig3/fig4
            for line in src.split("\n"):
                if "fig3" in line or "fig4" in line:
                    print(f"  Found: {repr(line)}")

import ast
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "def plot_solved_state_matrix_stats" in src:
        try:
            ast.parse(src)
            print("SYNTAX OK")
        except SyntaxError as e:
            print(f"SYNTAX ERROR: {e}")
        break

if func_patched and callsite_patched:
    with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Saved.")
else:
    print(f"NOT SAVED — func_patched={func_patched}, callsite_patched={callsite_patched}")
