import json, os, re

root = r'c:\git_repos\Graph-Neural-Networks'
canon_path = os.path.join(root, 'GNN_Powerflow_V2.7_Training.ipynb')
sess_dir = os.path.join(root, 'run_in_parallell', 'S_sessions')


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cell_src(c):
    return ''.join(c.get('source', []))


out = []

canon = load(canon_path)
canon_cells = canon['cells']
canon_srcs = [cell_src(c) for c in canon_cells]

# Which var name does canonical's OWN sweep cells (26-35) use?
out.append('--- canonical sweep cells (26-35) referencing networks_mixed_1500_* ---')
for i in range(26, 36):
    s = canon_srcs[i]
    names = set(re.findall(r'networks_mixed_1500_\w+', s))
    out.append(f'  cell {i}: {names}')

files = sorted(os.listdir(sess_dir))
out.append('')
out.append('--- per-file cell 2 (paths) active DATA_ROOT line ---')
for fname in files:
    nb = load(os.path.join(sess_dir, fname))
    cells = nb['cells']
    s2 = cell_src(cells[2])
    active_lines = [l for l in s2.splitlines() if l.strip().startswith('DATA_ROOT') and not l.strip().startswith('#DATA_ROOT')]
    out.append(f'{fname}: {active_lines}')

out.append('')
out.append('--- per-file cell 22 (dataset load) full text + which var name used in unique sweep cells ---')
for fname in files:
    nb = load(os.path.join(sess_dir, fname))
    cells = nb['cells']
    s22 = cell_src(cells[22])
    out.append(f'{fname} cell22: {s22!r}')
    # scan all cells for networks_mixed_1500_* usage
    names_used = set()
    for c in cells:
        names_used |= set(re.findall(r'networks_mixed_1500_\w+', cell_src(c)))
    out.append(f'   all networks_mixed_1500_* names referenced anywhere in file: {names_used}')

with open(os.path.join(root, '_audit_081_varnames.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
