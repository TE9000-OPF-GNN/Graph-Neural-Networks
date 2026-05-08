"""Patch col7 in PowerFlowDataset.__getitem__:
1. p_net_balance = -x_p[pq_mask].sum() instead of p_bus.sum()
2. Broadcast equally to all gen buses (no p_nom weighting)
"""
import json

path = 'GNN_Powerflow_V2.6_Training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    # Replacement 1: comment lines + p_net_balance computation
    '        # ── Step 1: p_net_balance — graph-level scalar for slack signal ───────\n':
    '        # ── Step 1: p_net_balance — total PQ load (known inputs only) ─────────\n',

    '        # Net active power balance = sum of all bus injections at this snapshot.\n':
    '        # Sum of PQ bus active power injections (loads). This is a pre-solve\n',

    '        # This gives the slack bus a varying signal correlated with its true P.\n':
    '        # quantity that correlates with P_slack without leaking the answer.\n',

    '        p_net_balance = float(p_bus.sum().item())\n':
    '        p_net_balance = -float(x_p[pq_mask].sum().item())\n',

    # Replacement 2: use_pnom_share block — replace comments
    '            # Participation-weighted balance: (p_nom_i / sum_p_nom) * p_net_balance\n':
    '            # Broadcast total PQ load equally to all generator (slack+PV) buses.\n',

    '            # Gives each generator its expected MW contribution to the total imbalance.\n':
    '            # No p_nom weighting — the GNN learns each bus\'s participation from\n',

    '            # For PQ nodes: 0.0 (they do not participate in slack compensation).\n':
    '            # cross-snapshot variation (which bus P actually responds to load changes).\n',
}

# Lines to DELETE from the source array
delete_lines = {
    '            # NOTE: use_pnet_balance flag is superseded — use_pnom_share controls this.\n',
    '            p_noms = network.generators["p_nom"].values.astype(float)\n',
    '            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0\n',
    '                    share = gen["p_nom"] / total_p_nom\n',
}

# Lines to RENAME (variable name change)
rename_lines = {
    '            p_nom_col = torch.zeros(n_buses, 1)\n':
    '            p_bal_col = torch.zeros(n_buses, 1)\n',

    '                    p_nom_col[bus_to_i[b], 0] += share * p_net_balance\n':
    '                    p_bal_col[bus_to_i[b], 0] = p_net_balance\n',

    '            x = torch.cat([x, p_nom_col], dim=1)\n':
    '            x = torch.cat([x, p_bal_col], dim=1)\n',
}

count = 0
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    src = cell['source']
    new_src = []
    for line in src:
        if line in delete_lines:
            count += 1
            continue  # skip this line
        if line in replacements:
            new_src.append(replacements[line])
            count += 1
        elif line in rename_lines:
            new_src.append(rename_lines[line])
            count += 1
        else:
            new_src.append(line)
    cell['source'] = new_src

print(f"Applied {count} line-level changes")
assert count == 14, f"Expected 14 changes, got {count}"

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patch applied successfully")
