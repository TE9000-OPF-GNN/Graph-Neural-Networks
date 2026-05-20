import json

nb = json.load(open('GNN_Powerflow_V2.6.1_Training.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'plot' in src.lower() or 'fig' in src or 'plt.' in src:
        first = c['source'][0].strip()[:80] if c['source'] else 'EMPTY'
        ct = c["cell_type"]
        print(f"Cell {i} ({ct}): {first}")
