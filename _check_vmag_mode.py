"""End-to-end vmag_mode diagnostic: load model, build dataset, run forward, check output."""
import os, sys, pickle, json
sys.path.insert(0, r'c:\git_repos\Graph-Neural-Networks')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pypsa
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import GATv2Conv, TransformerConv
from collections import deque
import logging

# ─── Load ALL code from Analysis notebook cells 9-12 (class + dataset + prediction) ───
with open(r'c:\git_repos\Graph-Neural-Networks\GNN_Powerflow_V2.6_Analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Need: PowerFlowGNN (cell 9), PowerFlowDataset (cell 10 or 11), predict helpers (cell 12)
# First, find which cells define what we need
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'class PowerFlowDataset' in src:
        print(f"PowerFlowDataset in cell {i}")
        exec(compile(src, f'<cell_{i}>', 'exec'))
    if 'class PowerFlowGNN' in src:
        print(f"PowerFlowGNN in cell {i}")
        exec(compile(src, f'<cell_{i}>', 'exec'))

# ─── Load model from pickle ───
TRAINING_RESULTS_DIR = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN\training_results_saved'
pkl_path = os.path.join(TRAINING_RESULTS_DIR, 'sweep_S3_vmag_residual.json')

with open(pkl_path, 'rb') as f:
    runs = pickle.load(f)

r = runs[0]
model = r['model']
model.eval()
print(f"\nModel: vmag_mode={model.vmag_mode!r}, head_mode={model.head_mode!r}, angle_mode={model.angle_mode!r}")

# ─── Load a test network ───
TRAINING_NETWORKS_DIR = r'C:\Users\STSI\OneDrive - USN\Data_PF_GNN\training_networks_saved'
# Find a network file
net_files = [f for f in os.listdir(TRAINING_NETWORKS_DIR) if 'mixed' in f.lower() and '1500' in f]
print(f"Available network files with 'mixed' and '1500': {net_files[:5]}")

if net_files:
    # Networks are stored as directories of .nc files
    net_dir = os.path.join(TRAINING_NETWORKS_DIR, 'mixed_1500_wide')
    nc_files = sorted([f for f in os.listdir(net_dir) if f.endswith('.nc')])
    print(f"Found {len(nc_files)} network files in mixed_1500_wide/")
    
    # Load network #36
    net_file = os.path.join(net_dir, nc_files[36])
    test_network = pypsa.Network()
    test_network.import_from_netcdf(net_file)
    print(f"Loaded network from {nc_files[36]}")
    # test_network already loaded above from nc file
    
    # Run power flow to get ground truth
    test_network.pf(use_seed=True, distribute_slack=False)
    true_vmag = test_network.buses_t.v_mag_pu.iloc[0].values
    print(f"\nGround truth vmag (first snapshot): {true_vmag[:5]} (mean={true_vmag.mean():.4f})")
    
    # Build dataset
    _angle_mode = getattr(model, "angle_mode", "node")
    dataset = PowerFlowDataset(
        [test_network],
        use_edge_features=True,
        use_pnom_share=(model.node_embedding.in_features >= 8),
        angle_mode=_angle_mode,
        ptdf_branch_mode="all" if _angle_mode == "edge_delta" else "lines",
    )
    
    data = dataset[0]
    print(f"\nDataset data: x.shape={data.x.shape}, edge_index.shape={data.edge_index.shape}")
    print(f"  pq_mask: {data.pq_mask}")
    print(f"  pv_mask: {data.pv_mask}")
    print(f"  slack_mask: {data.slack_mask}")
    print(f"  pq_mask.any(): {data.pq_mask.bool().any()}")
    print(f"  Has bfs_edge_idx: {hasattr(data, 'bfs_edge_idx')}")
    
    # Pad edge_attr if needed
    if hasattr(model, 'convs') and model.convs and hasattr(model.convs[0], 'lin_edge') and model.convs[0].lin_edge is not None:
        _expected = model.convs[0].lin_edge.in_channels
        if data.edge_attr.size(1) < _expected:
            data.edge_attr = torch.cat([data.edge_attr, torch.zeros(data.edge_attr.size(0), _expected - data.edge_attr.size(1))], dim=1)
            print(f"  Padded edge_attr to {data.edge_attr.shape}")
    
    # ─── RUN FORWARD ───
    print("\n" + "="*60)
    print("RUNNING MODEL FORWARD")
    print("="*60)
    
    with torch.no_grad():
        out = model(data)
    
    node_pred = out[0]
    print(f"node_pred shape: {node_pred.shape}")
    print(f"node_pred[:, 0] (Vmag col): {node_pred[:5, 0].numpy()}")
    
    # For PQ buses specifically
    pq_mask = data.pq_mask.bool()
    pq_vmag = node_pred[pq_mask, 0].numpy()
    print(f"\nPQ bus vmag predictions: {pq_vmag[:5]}")
    print(f"PQ vmag mean: {pq_vmag.mean():.4f}")
    print(f"PQ vmag std: {pq_vmag.std():.4f}")
    
    # Compare with ground truth for PQ buses
    buses = list(test_network.buses.index)
    pq_buses_idx = [i for i, b in enumerate(buses) if pq_mask[i]]
    true_pq_vmag = true_vmag[pq_buses_idx]
    print(f"\nTrue PQ vmag: {true_pq_vmag[:5]}")
    print(f"True PQ vmag mean: {true_pq_vmag.mean():.4f}")
    
    error = np.abs(pq_vmag - true_pq_vmag)
    print(f"\n|Predicted - True| for PQ buses: {error[:5]}")
    print(f"Mean absolute error: {error.mean():.4f}")
    
    if error.mean() > 0.5:
        print("\n*** ERROR IS LARGE! The +1.0 correction is NOT being applied ***")
        print(f"Expected vmag ≈ {true_pq_vmag.mean():.3f}, got ≈ {pq_vmag.mean():.3f}")
        if pq_vmag.mean() < 0.1:
            print("Predictions are ≈ 0 → +1.0 is NOT added in forward()")
        elif pq_vmag.mean() > 1.5:
            print("Predictions are ≈ 2 → +1.0 is added TWICE (double correction)")
    else:
        print("\n*** Predictions are correct — error is small ***")


