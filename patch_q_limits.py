"""
Patch: add q_min_pu / q_max_pu limits to generator addition in load_system_from_csv.
Cell 20 (index 19) in GNN_Powerflow_V2.6_DataGen.ipynb.
"""
import json, re

NB = r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_DataGen.ipynb'

OLD = (
    '        n.add("Generator", gen_id, bus=bus_name,\n'
    '              p_nom=p_nom,\n'
    '              p_min_pu=p_min / max(p_nom, 1e-3),\n'
    '              p_set=p_set, q_set=q_set, control=ctrl)'
)

NEW = (
    '        # Read Q limits (CSV stores physical MVAr; convert to pu of p_nom)\n'
    '        q_min_raw = float(row.get("min_q_at_target_p", row.get("min_q", 0.0)) or 0.0)\n'
    '        q_max_raw = float(row.get("max_q_at_target_p", row.get("max_q", 0.0)) or 0.0)\n'
    '        gen_kw = dict(bus=bus_name, p_nom=p_nom,\n'
    '                      p_min_pu=p_min / max(p_nom, 1e-3),\n'
    '                      p_set=p_set, q_set=q_set, control=ctrl)\n'
    '        # Q limits: only for non-slack PV buses with non-trivially-zero CSV entries\n'
    '        if ctrl != "Slack" and (q_min_raw != 0.0 or q_max_raw != 0.0):\n'
    '            gen_kw["q_min_pu"] = q_min_raw / max(p_nom, 1e-3)\n'
    '            gen_kw["q_max_pu"] = q_max_raw / max(p_nom, 1e-3)\n'
    '        n.add("Generator", gen_id, **gen_kw)'
)

with open(NB, encoding='utf-8', newline='') as f:
    raw = f.read()

# The notebook stores source as JSON strings, so we need to match the JSON-encoded form.
# Approach: load JSON, find cell 19, patch source as a plain string.
nb = json.loads(raw)
cell = nb['cells'][19]
src = ''.join(cell['source'])

count = src.count(OLD.replace('\\n', '\n'))
# OLD contains literal \n already (Python string), let's count directly
count = src.count(OLD)
print(f"Occurrences of OLD pattern: {count}")

if count != 1:
    print("ERROR: expected exactly 1 occurrence. Aborting.")
    raise SystemExit(1)

new_src = src.replace(OLD, NEW, 1)
print(f"Replacement applied. New occurrence count of OLD: {new_src.count(OLD)}")
print(f"Occurrence count of NEW: {new_src.count('q_min_raw')}")

# Split back into source lines list (each line ends with \n except the last)
lines = new_src.split('\n')
source_list = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
cell['source'] = source_list

with open(NB, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Patch written successfully.")
