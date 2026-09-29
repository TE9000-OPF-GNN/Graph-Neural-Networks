import json, os

root = r'c:\git_repos\Graph-Neural-Networks'
canon_path = os.path.join(root, 'GNN_Powerflow_V2.7_Training.ipynb')
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


canon = load(canon_path)
canon_cells = canon['cells']
canon_srcs = [cell_src(c) for c in canon_cells]

print('--- CANONICAL cell index : type : first line ---')
for i, c in enumerate(canon_cells):
    ctype = c['cell_type']
    s = canon_srcs[i]
    print('%3d %8s %s' % (i, ctype, first_line(s)))

print()
print('=' * 100)
