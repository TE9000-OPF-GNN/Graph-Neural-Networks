import json, os

root = r'c:\git_repos\Graph-Neural-Networks'
sess_dir = os.path.join(root, 'run_in_parallell', 'S_sessions')
canon_path = os.path.join(root, 'GNN_Powerflow_V2.7_Training.ipynb')


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cell_src(c):
    return ''.join(c.get('source', []))


out = []

files = ['GNN_Powerflow_V2.7_Training_d0.ipynb', 'GNN_Powerflow_V2.7_Training_d1.ipynb',
         'GNN_Powerflow_V2.7_Training_d2.ipynb', 'GNN_Powerflow_V2.7_Training_d3.ipynb',
         'GNN_Powerflow_V2.7_Training_d4.ipynb', 'GNN_Powerflow_V2.7_Training_d5.ipynb',
         'GNN_Powerflow_V2.7_Training_d6.ipynb', 'GNN_Powerflow_V2.7_Training_d7.ipynb']

out.append('--- session label (markdown cell 25) for each dX file ---')
srcs_by_file = {}
for fname in files:
    nb = load(os.path.join(sess_dir, fname))
    cells = nb['cells']
    md25 = cell_src(cells[25]).strip().splitlines()[0]
    out.append(f'{fname}: {md25}')
    srcs_by_file[fname] = [cell_src(c) for c in cells]

out.append('')
out.append('--- are cells [2,13,15,22] byte-identical across ALL d0..d7? ---')
for idx in [2, 13, 15, 22]:
    vals = set(srcs_by_file[f][idx] for f in files)
    out.append(f'cell index {idx}: distinct variants across d0-d7 = {len(vals)}')

# Now compare dn and AllSessions cells [2,13,15,22] to d0's version
dn = load(os.path.join(sess_dir, 'GNN_Powerflow_V2.7_Training_dn.ipynb'))
allsess = load(os.path.join(sess_dir, 'GNN_Powerflow_V2.7_Training_AllSessions.ipynb'))
dn_srcs = [cell_src(c) for c in dn['cells']]
all_srcs = [cell_src(c) for c in allsess['cells']]

out.append('')
out.append('--- dn / AllSessions cells [2,13,15,22] vs d0 same-index cells ---')
for idx in [2, 13, 15, 22]:
    out.append(f'cell {idx}: dn==d0? {dn_srcs[idx] == srcs_by_file["GNN_Powerflow_V2.7_Training_d0.ipynb"][idx]}; '
               f'AllSessions==d0? {all_srcs[idx] == srcs_by_file["GNN_Powerflow_V2.7_Training_d0.ipynb"][idx]}')

# Check dn's Sx sections (25,27,29,...) vs AllSessions equivalents (shifted by 1 due to extra blank md cell)
out.append('')
out.append('--- dn sweep-section cells (25..44) vs AllSessions sweep-section cells (26..45) [shifted by 1] ---')
dn_tail = dn_srcs[25:]
all_tail = all_srcs[26:]
out.append(f'dn tail len={len(dn_tail)}, AllSessions tail len={len(all_tail)}')
mismatches = [i for i in range(min(len(dn_tail), len(all_tail))) if dn_tail[i] != all_tail[i]]
out.append(f'mismatched offsets (0-based within tail): {mismatches}')

# Check whether dn/AllSessions Sx sections match each dX's own Sx unique section (25,26 in dX)
out.append('')
out.append('--- does dn/AllSessions embed the SAME text as each dX unique pair (25,26)? ---')
for i, fname in enumerate(files):
    pair = (srcs_by_file[fname][25], srcs_by_file[fname][26])
    in_dn = pair[0] in dn_tail and pair[1] in dn_tail
    in_all = pair[0] in all_tail and pair[1] in all_tail
    out.append(f'{fname} (S{i}) pair found in dn tail: {in_dn}; in AllSessions tail: {in_all}')

with open(os.path.join(root, '_audit_081_crosscheck.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
