import json, os

root = r'c:\git_repos\Graph-Neural-Networks'
sess_dir = os.path.join(root, 'run_in_parallell', 'S_sessions')


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cell_src(c):
    return ''.join(c.get('source', []))


def first_line(s, n=100):
    stripped = s.strip()
    if not stripped:
        return '(empty)'
    return stripped.splitlines()[0][:n]


out_lines = []
for fname in ['GNN_Powerflow_V2.7_Training_d0.ipynb', 'GNN_Powerflow_V2.7_Training_d7.ipynb',
              'GNN_Powerflow_V2.7_Training_dn.ipynb', 'GNN_Powerflow_V2.7_Training_AllSessions.ipynb']:
    path = os.path.join(sess_dir, fname)
    nb = load(path)
    cells = nb['cells']
    out_lines.append('=== %s ===' % fname)
    for i, c in enumerate(cells):
        out_lines.append('%3d %8s %s' % (i, c['cell_type'], first_line(cell_src(c))))
    out_lines.append('')

with open(os.path.join(root, '_audit_081_cells_dump.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print('done')
