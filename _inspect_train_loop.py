import json

with open('GNN_Powerflow_V2.7_Training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
src = ''.join(cells[14]['source'])
lines = src.split('\n')

# Find training loop - look for "for epoch" or "for batch"
for i, line in enumerate(lines):
    if 'for epoch' in line.lower() or 'for batch' in line:
        print(f"Found at line {i}: {line}")

# Also find phys_loss calls
for i, line in enumerate(lines):
    if 'physics_informed_loss_batch' in line:
        print(f"phys_loss at line {i}: {line.strip()}")

# Print the training loop section (around line 300-500)
print("\n\nLines 300-500:")
for i in range(300, min(500, len(lines))):
    print(f"{i:4d}: {lines[i]}")
