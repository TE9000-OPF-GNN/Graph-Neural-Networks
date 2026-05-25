"""
Integration test: Verify the full GNN model works with both angle modes.
Uses the actual notebook code by executing it via exec().
"""
import sys
sys.path.insert(0, '.')

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.data import Data, Batch

# Load notebook cells and extract the necessary source code
nb = json.load(open('GNN_Powerflow_V2.6.1_Training.ipynb', 'r', encoding='utf-8'))

# Find and compile the BFS utilities cell
bfs_src = None
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'def precompute_bfs_order' in src:
        bfs_src = src
        break

# Find and compile the model cell
model_src = None
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if 'class PowerFlowGNN' in src:
        model_src = src
        break

assert bfs_src is not None, "BFS cell not found"
assert model_src is not None, "Model cell not found"

# Setup namespace
from dataclasses import dataclass
from typing import List, Optional
import copy
import logging
import pandas as pd
logger = logging.getLogger(__name__)

ns = {
    'torch': torch,
    'nn': nn,
    'F': F,
    'np': np,
    'deque': deque,
    'dataclass': dataclass,
    'List': List,
    'Optional': Optional,
    'copy': copy,
    'logging': logging,
    'logger': logger,
    'pd': pd,
}

# Import required PyG modules
from torch_geometric.nn import GATv2Conv, TransformerConv, GCNConv, GraphConv
ns['GATv2Conv'] = GATv2Conv
ns['TransformerConv'] = TransformerConv
ns['GCNConv'] = GCNConv
ns['GraphConv'] = GraphConv
ns['Data'] = Data
ns['Batch'] = Batch

# Execute BFS utils
exec(bfs_src, ns)

# Execute model definition
exec(model_src, ns)

PowerFlowGNN = ns['PowerFlowGNN']
precompute_bfs_order = ns['precompute_bfs_order']
reconstruct_theta_from_delta = ns['reconstruct_theta_from_delta']

print("="*60)
print("INTEGRATION TEST: PowerFlowGNN with angle_mode")
print("="*60)

# ─── Create fake graph data ──────────────────────────────────────────────────
def make_test_data(num_nodes=5, num_fwd_edges=6, angle_mode="node"):
    """Create minimal fake Data object for testing."""
    # Bus types: 0=slack, 1=PV, 2,3,4=PQ
    x = torch.zeros(num_nodes, 7)
    x[0, 0] = 1.0  # slack
    x[1, 1] = 1.0  # PV
    x[2:, 2] = 1.0  # PQ
    x[0, 5] = 1.0; x[1, 5] = 1.02  # known Vmag
    x[2, 3] = -0.5; x[3, 3] = -0.3; x[4, 3] = -0.2  # PQ P
    x[2, 4] = -0.1; x[3, 4] = -0.05; x[4, 4] = -0.08  # PQ Q
    x[1, 3] = 0.8  # PV P
    
    # Forward edges: 0-1, 0-2, 1-3, 2-3, 2-4, 3-4
    fwd = [[0,1],[0,2],[1,3],[2,3],[2,4],[3,4]]
    # Bidirectional
    edges = []
    fwd_mask = []
    for f, t in fwd:
        edges.append([f, t])
        fwd_mask.append(True)
        edges.append([t, f])
        fwd_mask.append(False)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.randn(len(edges), 6)  # 6 edge features
    forward_edge_mask = torch.tensor(fwd_mask, dtype=torch.bool)
    
    # Masks
    slack_mask = torch.tensor([True, False, False, False, False])
    pv_mask = torch.tensor([False, True, False, False, False])
    pq_mask = torch.tensor([False, False, True, True, True])
    
    # Target
    y = torch.randn(num_nodes, 4)
    y[:, 0] = y[:, 0].abs() + 0.9  # Vmag near 1.0
    
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.slack_mask = slack_mask
    data.pv_mask = pv_mask
    data.pq_mask = pq_mask
    data.forward_edge_mask = forward_edge_mask
    data.network_idx = torch.tensor([0], dtype=torch.long)
    
    if angle_mode == "edge_delta":
        # BFS metadata
        edge_index_fwd = edge_index[:, forward_edge_mask]
        theta_true = y[:, 1]
        y_delta_theta = theta_true[edge_index_fwd[0]] - theta_true[edge_index_fwd[1]]
        data.y_delta_theta = y_delta_theta
        
        slack_idx = 0
        bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
            edge_index_fwd, slack_idx, num_nodes
        )
        data.bfs_edge_idx = bfs_edge_idx
        data.bfs_signs = bfs_signs
        data.bfs_node_order = bfs_node_order
        data.bfs_parent = bfs_parent
    
    return data


# ─── Test A: angle_mode="node" (default, backward-compatible) ────────────────
print("\nTest A: angle_mode='node' (backward-compat)...")

model_node = PowerFlowGNN(
    node_features=7, edge_features=6, hidden_dim=32, num_layers=2,
    head_mode="standard", angle_mode="node"
)
data_node = make_test_data(angle_mode="node")

# Test forward without return_embeddings
out = model_node(data_node)
assert len(out) == 3, f"Expected 3-tuple, got {len(out)}"
node_pred, ptdf_pred, dtp = out
assert node_pred.shape == (5, 4), f"Expected [5,4], got {node_pred.shape}"
assert dtp is None, f"Expected None for delta_theta_pred in node mode, got {type(dtp)}"
print(f"  Forward (no embed): node_pred={node_pred.shape}, ptdf_pred={ptdf_pred.shape}, delta_theta_pred=None ✓")

# Test with return_embeddings
out_emb = model_node(data_node, return_embeddings=True)
assert len(out_emb) == 4, f"Expected 4-tuple, got {len(out_emb)}"
np_emb, h_n, h_e, dtp_emb = out_emb
assert np_emb.shape == (5, 4)
assert dtp_emb is None
print(f"  Forward (embed): node_pred={np_emb.shape}, h_nodes={h_n.shape}, h_edges={h_e.shape}, delta_theta=None ✓")


# ─── Test B: angle_mode="edge_delta", standard heads ─────────────────────────
print("\nTest B: angle_mode='edge_delta', head_mode='standard'...")

model_edge = PowerFlowGNN(
    node_features=7, edge_features=6, hidden_dim=32, num_layers=2,
    head_mode="standard", angle_mode="edge_delta"
)
data_edge = make_test_data(angle_mode="edge_delta")

# Forward without embeddings
out = model_edge(data_edge)
assert len(out) == 3, f"Expected 3-tuple, got {len(out)}"
node_pred, ptdf_pred, dtp = out
assert node_pred.shape == (5, 3), f"Expected [5,3], got {node_pred.shape}"
assert dtp is not None, "Expected delta_theta_pred tensor"
assert dtp.shape == (6,), f"Expected [6] (num_fwd_edges), got {dtp.shape}"
print(f"  Forward (no embed): node_pred={node_pred.shape}, delta_theta={dtp.shape} ✓")

# Forward with embeddings
out_emb = model_edge(data_edge, return_embeddings=True)
np_emb, h_n, h_e, dtp_emb = out_emb
assert np_emb.shape == (5, 3)
assert dtp_emb.shape == (6,)
print(f"  Forward (embed): node_pred={np_emb.shape}, delta_theta={dtp_emb.shape} ✓")


# ─── Test C: angle_mode="edge_delta", with_encoder heads ─────────────────────
print("\nTest C: angle_mode='edge_delta', head_mode='with_encoder'...")

model_enc = PowerFlowGNN(
    node_features=7, edge_features=6, hidden_dim=32, num_layers=2,
    head_mode="with_encoder", angle_mode="edge_delta"
)

out = model_enc(data_edge)
node_pred, _, dtp = out
assert node_pred.shape == (5, 3), f"Expected [5,3], got {node_pred.shape}"
assert dtp.shape == (6,)
print(f"  Forward: node_pred={node_pred.shape}, delta_theta={dtp.shape} ✓")


# ─── Test D: Gradient flow end-to-end ────────────────────────────────────────
print("\nTest D: End-to-end gradient flow through model + reconstruction...")

model_edge.train()
node_pred, _, _, dtp = model_edge(data_edge, return_embeddings=True)

# Reconstruct θ
theta_recon = reconstruct_theta_from_delta(
    dtp, data_edge.bfs_edge_idx, data_edge.bfs_signs,
    data_edge.bfs_node_order, data_edge.bfs_parent, 5
)

# Loss on θ
loss = F.mse_loss(theta_recon, data_edge.y[:, 1])
loss.backward()

# Check gradients flow to edge_angle_pred
grad_ok = model_edge.edge_angle_pred.weight.grad is not None
grad_nonzero = grad_ok and model_edge.edge_angle_pred.weight.grad.abs().sum() > 0
assert grad_ok, "No gradient on edge_angle_pred.weight!"
assert grad_nonzero, "Gradient is zero on edge_angle_pred.weight!"
print(f"  PASS: Gradients flow through reconstruction → edge_angle_pred (grad_norm={model_edge.edge_angle_pred.weight.grad.norm():.4f})")


# ─── Test E: Batching with exclude_keys ──────────────────────────────────────
print("\nTest E: Batching multiple graphs with BFS exclude_keys...")

data1 = make_test_data(angle_mode="edge_delta")
data2 = make_test_data(angle_mode="edge_delta")

# Simulate collate: remove BFS before batching
bfs_keys = ['bfs_edge_idx', 'bfs_signs', 'bfs_node_order', 'bfs_parent', 'y_delta_theta']
bfs_lists = {k: [] for k in bfs_keys}
for d in [data1, data2]:
    for k in bfs_keys:
        if hasattr(d, k):
            bfs_lists[k].append(getattr(d, k))
            delattr(d, k)

batch = Batch.from_data_list([data1, data2])

# Restore as lists
for k, v in bfs_lists.items():
    setattr(batch, k, v)

# Forward on batch
model_edge.zero_grad()
node_pred_b, _, _, dtp_b = model_edge(batch, return_embeddings=True)
assert node_pred_b.shape == (10, 3), f"Expected [10,3] for batch of 2, got {node_pred_b.shape}"
assert dtp_b.shape == (12,), f"Expected [12] (2*6 fwd edges), got {dtp_b.shape}"

# Per-graph reconstruction
dt_sizes = [t.size(0) for t in bfs_lists['y_delta_theta']]
dt_offsets = [0] + [sum(dt_sizes[:i+1]) for i in range(len(dt_sizes))]
node_batch = batch.batch
num_graphs = 2

for g in range(num_graphs):
    nm = (node_batch == g)
    n_i = int(nm.sum().item())
    dtp_g = dtp_b[dt_offsets[g]:dt_offsets[g+1]]
    theta_g = reconstruct_theta_from_delta(
        dtp_g, bfs_lists['bfs_edge_idx'][g], bfs_lists['bfs_signs'][g],
        bfs_lists['bfs_node_order'][g], bfs_lists['bfs_parent'][g], n_i
    )
    assert theta_g.shape == (n_i,), f"Graph {g}: expected [{n_i}], got {theta_g.shape}"

print(f"  PASS: Batched forward + per-graph reconstruction works correctly")

# Restore data for future use
for i, d in enumerate([data1, data2]):
    for k in bfs_keys:
        setattr(d, k, bfs_lists[k][i])


print("\n" + "="*60)
print("ALL INTEGRATION TESTS PASSED")
print("="*60)
