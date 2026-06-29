from pathlib import Path

p = Path("GNN_Powerflow_V2.7_Analysis.ipynb")
t = p.read_text(encoding="utf-8")

new_block = '''    "    if max_w_flow is not None:\\n",\n    "        for _, ax, title in _eff_panels[2:]:  # flow panels get cap line\\n",\n    "            ax.axhline(max_w_flow, color=\\"red\\", linestyle=\\"--\\", alpha=0.4,\\n",\n    "                        label=f\\"max_w_flow={max_w_flow}\\")\\n",\n    "\\n",\n    "    # [STSI 250626] Smart y-axis scaling when flow weights are tiny vs cap\\n",\n    "    for i, (key, ax, title) in enumerate(_eff_panels):\\n",\n    "        if i >= 2:  # flow panels only (indices 2-5)\\n",\n    "            all_values = []\\n",\n    "            for r in runs:\\n",\n    "                h = r.get(\\"history\\", {})\\n",\n    "                if key in h and h[key]:\\n",\n    "                    all_values.extend([v for v in h[key] if isinstance(v, (int, float)) and v > 0])\\n",\n    "            if all_values and max_w_flow is not None:\\n",\n    "                max_data = max(all_values)\\n",\n    "                if max_data > 0 and (max_w_flow / max_data) > 1000:\\n",\n    "                    ax.set_yscale(\\"symlog\\", linthresh=1e-8)\\n",\n    "                    ax.set_ylabel(\\"Effective Weight (log scale)\\")\\n",\n    "\\n",\n    "    for _, ax, title in _eff_panels:\\n",\n    "        ax.set_xlabel(\\"Epoch\\")\\n",\n    "        if ax.get_ylabel() != \\\"Effective Weight (log scale)\\\":\\n",\n    "            ax.set_ylabel(\\"Effective Weight\\")\\n",\n    "        ax.set_title(f\\"{title_prefix} — {title}\\")\\n",\n    "        ax.grid(True, alpha=0.3)\\n",\n'''

old_block = '''    "    if max_w_flow is not None:\\n",\n    "        for _, ax, title in _eff_panels[2:]:  # flow panels get cap line\\n",\n    "            ax.axhline(max_w_flow, color=\\"red\\", linestyle=\\"--\\", alpha=0.4,\\n",\n    "                        label=f\\"max_w_flow={max_w_flow}\\")\\n",\n    "\\n",\n    "    for _, ax, title in _eff_panels:\\n",\n    "        ax.set_xlabel(\\"Epoch\\")\\n",\n    "        ax.set_ylabel(\\"Effective Weight\\")\\n",\n    "        ax.set_title(f\\"{title_prefix} — {title}\\")\\n",\n    "        ax.grid(True, alpha=0.3)\\n",\n'''

if new_block not in t:
    raise SystemExit("patched block not found; nothing reverted")

t = t.replace(new_block, old_block, 1)
p.write_text(t, encoding="utf-8")
print("reverted")
