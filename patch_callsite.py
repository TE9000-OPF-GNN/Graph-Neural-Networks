import json

NOTEBOOK = "GNN_Powerflow_V2.6_DataGen.ipynb"
nb = json.load(open(NOTEBOOK, encoding="utf-8"))

for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "plot_topology_stats" in src and "all_networks" in src and "fig2" in src:
        old = "fig2 = plot_electrical_params(all_networks,clip_percentile=(1, 95), n_bins=n_bins)\nplt.show()"
        new = (
            "fig2 = plot_electrical_params(all_networks, clip_percentile=(1, 95), n_bins=n_bins)\n"
            "plt.show()\n"
            "fig2b = plot_electrical_params_stats(all_networks)  "
            "# [STSI 210526]:mean\u00b1std comparison alongside histograms\n"
            "plt.show()"
        )
        if old in src:
            cell["source"] = [src.replace(old, new, 1)]
            print("Patched call-site cell")
        else:
            print("WARNING: exact string not found, trying partial match...")
            # Try without spaces
            for line in src.split("\n"):
                if "plot_electrical_params" in line and "clip_percentile" in line:
                    print(f"  Found: {repr(line)}")
        break

with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved")
