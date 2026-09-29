import json, os, difflib

root = r'c:\git_repos\Graph-Neural-Networks'
canon_path = os.path.join(root, 'GNN_Powerflow_V2.7_Training.ipynb')
d0_path = os.path.join(root, 'run_in_parallell', 'S_sessions', 'GNN_Powerflow_V2.7_Training_d0.ipynb')


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cell_src(c):
    return ''.join(c.get('source', []))


canon = load(canon_path)
d0 = load(d0_path)

canon_cells = canon['cells']
d0_cells = d0['cells']

out = []
for idx in [2, 13, 15, 22]:
    c_src = cell_src(canon_cells[idx])
    d_src = cell_src(d0_cells[idx])
    out.append('=' * 30 + f' CELL INDEX {idx} ' + '=' * 30)
    diff = difflib.unified_diff(
        c_src.splitlines(keepends=True),
        d_src.splitlines(keepends=True),
        fromfile=f'canonical[{idx}]',
        tofile=f'd0[{idx}]',
    )
    out.extend(diff)
    out.append('')

with open(os.path.join(root, '_audit_081_diff_cells.txt'), 'w', encoding='utf-8') as f:
    f.writelines(out if isinstance(out[0], str) and out[0].endswith('\n') else [l + '\n' if not l.endswith('\n') else l for l in out])
print('done')
