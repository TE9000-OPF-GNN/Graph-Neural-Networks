import json

NB = r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6.2_Training.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'def run_hparam_sweep(' in src:
        lines = cell['source']
        for j, l in enumerate(lines):
            if '"ptdf_branch_mode": pcfg.ptdf_branch_mode,' in l:
                if j + 1 < len(lines) and 'ptdf_changepoint' in lines[j + 1]:
                    print('Already present')
                else:
                    lines.insert(j + 1, '            "ptdf_changepoint": pcfg.ptdf_changepoint,  # [STSI 290526]\n')
                    print(f'Inserted after L{j}')
                break
        break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done')
