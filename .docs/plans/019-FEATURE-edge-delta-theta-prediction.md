---
type: FEATURE
status: done
priority: High
effort: 4h
labels: [model-architecture, training, GNN]
depends-on: []
created: 2026-05-25
completed: 2026-05-26
summary: "Implemented edge-based Δθ prediction with BFS reconstruction. All tests pass."
---

# Edge-Based Δθ Prediction (angle_mode="edge_delta")

## Problem

The GNN currently predicts absolute voltage angle θ per node (referenced to slack, θ_slack=0). For buses far from the slack, θ accumulates along paths — making it a **global** quantity that local message passing is inherently bad at learning. This leads to higher angle errors on distant buses and requires message-passing depth exceeding graph diameter.

Predicting Δθ = θ_from − θ_to **per edge** is locally determined by branch impedance and power flow — exactly what message passing excels at. Literature confirms this: Hansen et al. (2022) demonstrate branch-level prediction in power GNNs; Farivar & Low (2013) provide the theoretical foundation (Branch Flow Model).

## Solution

Add `angle_mode="edge_delta"` to `PowerFlowGNN`. In this mode:

1. **Node heads** predict 3 quantities: [Vmag, P, Q] (angle removed from node prediction)
2. **New edge head** predicts Δθ per forward edge (1 value per branch)
3. **θ reconstruction** via pre-computed BFS traversal from slack (differentiable, O(N))
4. **Output shape unchanged**: final evaluation receives `[N, 4]` = [Vmag, Vang, P, Q] — backward-compatible with all analysis/eval code

**Why Option C (predict Δθ + reconstruct θ) instead of Option B (dual node+edge)?**
- Single source of truth (Δθ) — no consistency loss hyperparameter
- Directly solves the accumulation problem (not a soft constraint)
- Δθ MSE is physically meaningful (uniform scale, drives line flow)
- Physics loss still works via reconstructed θ → existing Y-bus code unchanged

**Why θ reconstruction is OUTSIDE forward()?**
- Training uses batch_size=32-64. BFS metadata (bfs_order, bfs_signs) is per-graph with variable N → must go in `exclude_keys` → arrives as list, not stacked tensor
- Reconstruction happens in the per-graph loss loop (same pattern as physics/PTDF loss)
- Same pattern in evaluation: per-graph reconstruction + timing

**Sign convention**: `Δθ_e = θ_from − θ_to` (from=bus0, to=bus1). Positive Δθ → power flows bus0→bus1. Consistent with `forward_edge_mask`, `y_line_p` (p0), and PTDF sign.

## Scope

**Included:**
- `angle_mode` parameter in `PowerFlowGNN.__init__` and `forward()`
- Modified node heads for both `standard` and `with_encoder` head modes
- New `edge_angle_pred` Linear head on forward edge embeddings
- BFS-order precomputation in `PowerFlowDataset._create_graph_data`
- `y_delta_theta` target stored per Data object
- Per-graph θ reconstruction in training loop (physics loss integration)
- Per-graph θ reconstruction in `evaluate_gnn_on_test_set`
- `theta_reconstruction_time_ms` metric in evaluation output
- Updated `collate_with_ptdf` and `batched_gnn_forward` exclude_keys
- All existing hyperparameter sweep / `run_info` serialization supporting `angle_mode`
- Δθ MSE loss term (edge-level)

**Not Included:**
- Phase 2: edge-local physics loss reformulation (future task)
- ΔVmag prediction (no physical motivation)
- V2.7 DC baseline / residual prediction (orthogonal, composable later)
- Changes to Analysis notebook (backward-compatible output)
- Changes to DataGen notebook
- New loss weight hyperparameter for Δθ vs node MSE (use equal weighting initially)

## Affected Files

| File | Change |
|------|--------|
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~8 (dataset) | Add BFS precomputation, store `y_delta_theta`, `bfs_order`, `bfs_edge_idx`, `bfs_signs` on Data |
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~11 (model) | Add `angle_mode` param, edge_angle_pred head, modify node heads, change forward() return |
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~13 (training loop) | Add Δθ MSE loss, per-graph θ reconstruction for physics loss |
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~15 (collate) | Add BFS tensors to exclude_keys handling |
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~17 (evaluate) | Per-graph θ reconstruction, timing, [N,4] assembly |
| `GNN_Powerflow_V2.6.1_Training.ipynb` Cell ~18 (sweep) | Thread `angle_mode` through `run_hparam_sweep`, `make_run_key`, `run_info` |

## Implementation Steps

### 1. Add BFS Precomputation Utility Function

Create a utility function that computes BFS traversal order from the slack bus on a given edge_index. This is called once per graph during dataset construction.

```python
def precompute_bfs_order(edge_index_fwd, slack_idx, num_nodes):
    """
    Compute BFS traversal from slack for θ reconstruction from Δθ.
    
    Returns:
        bfs_edge_idx:   [N-1] indices into forward edges — edges in BFS tree order
        bfs_signs:      [N-1] float tensor — +1 or -1
        bfs_node_order: [N-1] node indices in BFS visit order (excludes slack)
        bfs_parent:     [N-1] parent node index for each BFS child
    
    Convention: θ_child = θ_parent - sign * Δθ_edge
        sign = +1 when edge is parent→child (forward direction = from→to)
        sign = -1 when edge is child→parent (we traverse against forward direction)
    """
    from collections import deque
    
    # Build undirected adjacency from forward edges
    # Each forward edge e: from=edge_index_fwd[0,e], to=edge_index_fwd[1,e]
    adj = [[] for _ in range(num_nodes)]
    n_edges = edge_index_fwd.size(1)
    for e in range(n_edges):
        u = edge_index_fwd[0, e].item()
        v = edge_index_fwd[1, e].item()
        adj[u].append((v, e, +1.0))   # forward direction: θ_v = θ_u - (+1)*Δθ_e
        adj[v].append((u, e, -1.0))   # reverse direction:  θ_u = θ_v - (-1)*Δθ_e
    
    visited = [False] * num_nodes
    visited[slack_idx] = True
    queue = deque([slack_idx])
    
    bfs_edge_idx = []
    bfs_signs = []
    bfs_node_order = []
    bfs_parent = []
    
    while queue:
        node = queue.popleft()
        for neighbor, edge_e, sign in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                bfs_edge_idx.append(edge_e)
                bfs_signs.append(sign)
                bfs_node_order.append(neighbor)
                bfs_parent.append(node)
                queue.append(neighbor)
    
    assert len(bfs_node_order) == num_nodes - 1, (
        f"BFS did not reach all nodes: visited {len(bfs_node_order)+1}/{num_nodes}"
    )
    
    return (
        torch.tensor(bfs_edge_idx, dtype=torch.long),
        torch.tensor(bfs_signs, dtype=torch.float32),
        torch.tensor(bfs_node_order, dtype=torch.long),
        torch.tensor(bfs_parent, dtype=torch.long),
    )
```

### 2. Add Differentiable θ Reconstruction Function

This function is used in both training (for physics loss) and evaluation. Sequential loop over N-1 BFS steps; each step is a single addition — fully differentiable via autograd. For N≤118, the loop is negligible vs. GNN forward.

```python
def reconstruct_theta_from_delta(delta_theta_fwd, bfs_edge_idx, bfs_signs, 
                                  bfs_node_order, bfs_parent, num_nodes):
    """
    Reconstruct absolute θ from edge Δθ predictions using pre-computed BFS order.
    Differentiable w.r.t. delta_theta_fwd (chain of additions).
    
    Args:
        delta_theta_fwd: [E_fwd] predicted Δθ per forward edge
        bfs_edge_idx:    [N-1] edge indices in BFS traversal order
        bfs_signs:       [N-1] signs (+1/-1)
        bfs_node_order:  [N-1] child node indices in BFS visit order
        bfs_parent:      [N-1] parent node index for each child
        num_nodes:       int, total nodes
    
    Returns:
        theta: [N] reconstructed absolute angles (θ_slack = 0)
    """
    theta = torch.zeros(num_nodes, device=delta_theta_fwd.device, dtype=delta_theta_fwd.dtype)
    
    for i in range(len(bfs_node_order)):
        child = bfs_node_order[i].item()
        parent_node = bfs_parent[i].item()
        e = bfs_edge_idx[i]
        s = bfs_signs[i]
        theta[child] = theta[parent_node] - s * delta_theta_fwd[e]
    
    return theta
```

**Autograd note**: The sequential loop creates a chain of N-1 dependent ops in the computation graph. For N=118 this is 117 additions — trivial overhead. Verify with `torch.autograd.gradcheck` on a small graph (see acceptance criteria).

### 3. Extend `_create_graph_data` to Store BFS Metadata + Δθ Target

In `PowerFlowDataset._create_graph_data`, after computing `edge_index` and `forward_edge_mask`:

```python
# --- Δθ target and BFS metadata (for angle_mode="edge_delta") ---
if self.angle_mode == "edge_delta":
    # Extract forward edge_index only
    edge_index_fwd = edge_index[:, forward_edge_mask]
    
    # Compute Δθ target: θ_from - θ_to for each forward edge
    theta_true = y[:, 1]  # absolute angles from power flow solution
    y_delta_theta = theta_true[edge_index_fwd[0]] - theta_true[edge_index_fwd[1]]
    data.y_delta_theta = y_delta_theta  # [E_fwd]
    
    # Precompute BFS order from slack
    slack_idx = int(slack_mask.nonzero(as_tuple=True)[0][0])
    bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent = precompute_bfs_order(
        edge_index_fwd, slack_idx, num_nodes
    )
    data.bfs_edge_idx = bfs_edge_idx
    data.bfs_signs = bfs_signs
    data.bfs_node_order = bfs_node_order
    data.bfs_parent = bfs_parent
```

`PowerFlowDataset.__init__` gets a new parameter:
```python
def __init__(self, ..., angle_mode: str = "node"):
    self.angle_mode = angle_mode
```

### 4. Modify `PowerFlowGNN` — Add `angle_mode` and Edge Angle Head

**`__init__` changes:**

```python
def __init__(self, ..., angle_mode: str = "node"):
    ...
    self.angle_mode = angle_mode
    
    if self.head_mode == "standard":
        if angle_mode == "edge_delta":
            # Node heads: Vmag, P, Q (3 outputs, no angle)
            self.vmag_pred = nn.Linear(hidden_dim, 1)
            self.p_pred = nn.Linear(hidden_dim, 1)
            self.q_pred = nn.Linear(hidden_dim, 1)
        else:
            # Existing: Vmag, Vang, P, Q (4 separate heads)
            self.vmag_pred = nn.Linear(hidden_dim, 1)
            self.vang_pred = nn.Linear(hidden_dim, 1)
            self.p_pred = nn.Linear(hidden_dim, 1)
            self.q_pred = nn.Linear(hidden_dim, 1)
    
    elif self.head_mode == "with_encoder":
        if angle_mode == "edge_delta":
            # PQ buses: predict Vmag only (1 output)
            self.pq_head = nn.Linear(hidden_dim, 1)
            # PV buses: predict Q only (1 output)
            self.pv_head = nn.Linear(hidden_dim, 1)
            # Slack buses: predict P, Q (2 outputs, unchanged)
            self.slack_head = nn.Linear(hidden_dim, 2)
        else:
            # Existing: PQ→[Vmag,Vang], PV→[Vang,Q], Slack→[P,Q]
            self.pq_head = nn.Linear(hidden_dim, 2)
            self.pv_head = nn.Linear(hidden_dim, 2)
            self.slack_head = nn.Linear(hidden_dim, 2)
    
    # Edge angle head (only for edge_delta mode)
    if angle_mode == "edge_delta":
        self.edge_angle_pred = nn.Linear(hidden_dim, 1)
```

**`forward()` changes:**

```python
def forward(self, data, return_embeddings=False):
    # ... (conv layers unchanged) ...
    h_nodes = h
    
    # --- Node predictions (without angle in edge_delta mode) ---
    if self.head_mode == "standard":
        if self.angle_mode == "edge_delta":
            vmag = self.vmag_pred(h_nodes)
            p = self.p_pred(h_nodes)
            q = self.q_pred(h_nodes)
            node_pred = torch.cat([vmag, p, q], dim=-1)  # [N, 3]
        else:
            # existing 4-column output
            node_pred = torch.cat([vmag, vang, p, q], dim=-1)  # [N, 4]
    
    elif self.head_mode == "with_encoder":
        if self.angle_mode == "edge_delta":
            # 3 columns: [Vmag, P, Q]
            node_pred = torch.zeros(h_nodes.size(0), 3, device=h_nodes.device)
            # PQ: predict Vmag (col 0)
            node_pred[pq_mask, 0] = self.pq_head(h_nodes[pq_mask]).squeeze(-1)
            # PV: predict Q (col 2)
            node_pred[pv_mask, 2] = self.pv_head(h_nodes[pv_mask]).squeeze(-1)
            # Slack: predict P (col 1), Q (col 2)
            slack_out = self.slack_head(h_nodes[slack_mask])
            node_pred[slack_mask, 1] = slack_out[:, 0]
            node_pred[slack_mask, 2] = slack_out[:, 1]
        else:
            # existing 4-column with_encoder logic
            ...
    
    # --- Edge embeddings (unchanged) ---
    row, col = data.edge_index
    h_src = h_nodes[row]
    h_dst = h_nodes[col]
    edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
    h_edges = self.edge_mlp(edge_input)  # [E_all, hidden_dim]
    
    # --- Edge angle prediction (edge_delta mode) ---
    delta_theta_pred = None
    if self.angle_mode == "edge_delta":
        fwd_mask = data.forward_edge_mask  # [E_all] bool
        h_edges_fwd = h_edges[fwd_mask]    # [E_fwd, hidden_dim]
        delta_theta_pred = self.edge_angle_pred(h_edges_fwd).squeeze(-1)  # [E_fwd]
    
    if return_embeddings:
        return node_pred, h_nodes, h_edges, delta_theta_pred
    
    # PTDF (unchanged)
    H_W = h_edges @ self.ptdf_W
    ptdf_pred = H_W @ h_nodes.T
    return node_pred, ptdf_pred, delta_theta_pred
```

**Key design choice**: `forward()` returns `delta_theta_pred` as a new third/fourth element. When `angle_mode="node"`, it's `None`. Callers unpack accordingly.

### 5. Update `collate_with_ptdf` and `batched_gnn_forward`

Add BFS tensors to the variable-size exclusion handling:

**In `collate_with_ptdf`:**

```python
# Before Batch.from_data_list, also remove BFS metadata:
bfs_keys = ['bfs_edge_idx', 'bfs_signs', 'bfs_node_order', 'bfs_parent', 
            'y_delta_theta']
bfs_data = {k: [] for k in bfs_keys}
for d in batch:
    for k in bfs_keys:
        if hasattr(d, k):
            bfs_data[k].append(getattr(d, k))
            delattr(d, k)

# ... existing Batch.from_data_list ...

# Reattach as lists:
for k, v_list in bfs_data.items():
    if v_list:
        setattr(batch_out, k, v_list)
```

**In `batched_gnn_forward`:**

```python
exclude_keys = ['y_ptdf', 'ptdf_line_index', 'y_line_p',
                'bfs_edge_idx', 'bfs_signs', 'bfs_node_order', 
                'bfs_parent', 'y_delta_theta']
```

### 6. Update Training Loop — Δθ MSE Loss + Physics Loss Integration

**Δθ MSE loss** (add after existing MSE computation):

```python
# --- Δθ edge MSE loss (angle_mode="edge_delta") ---
if model.angle_mode == "edge_delta" and delta_theta_pred is not None:
    # y_delta_theta is in exclude_keys → stored as list of per-graph tensors
    y_delta_theta_batch = torch.cat(batch.y_delta_theta, dim=0).to(device)
    loss_delta_theta = F.mse_loss(delta_theta_pred, y_delta_theta_batch)
else:
    loss_delta_theta = torch.tensor(0.0, device=device)
```

**Computing per-graph Δθ slices** — since `y_delta_theta` is in `exclude_keys`, each graph's forward edge count varies. Use cumulative sizes from the list:

```python
# Precompute per-graph Δθ slice boundaries (once before the per-graph loop):
if model.angle_mode == "edge_delta":
    delta_theta_sizes = [t.size(0) for t in batch.y_delta_theta]  # E_fwd per graph
    delta_theta_offsets = [0] + list(itertools.accumulate(delta_theta_sizes))
```

**Physics loss integration** — extend the existing per-graph physics loop:

```python
# Inside physics_informed_loss_batch, per-graph loop (graph index i):
if model.angle_mode == "edge_delta":
    # Slice this graph's Δθ predictions from the concatenated batch output
    dt_start = delta_theta_offsets[i]
    dt_end = delta_theta_offsets[i + 1]
    delta_theta_graph = delta_theta_pred[dt_start:dt_end]  # [E_fwd_i]
    
    theta_reconstructed = reconstruct_theta_from_delta(
        delta_theta_graph,
        batch.bfs_edge_idx[i],
        batch.bfs_signs[i],
        batch.bfs_node_order[i],
        batch.bfs_parent[i],
        num_nodes_i,
    )
    
    # Assemble full [N_i, 4] for physics loss: [Vmag, θ_reconstructed, P, Q]
    node_pred_full = torch.zeros(num_nodes_i, 4, device=device)
    node_pred_full[:, 0] = node_pred_partial[node_slice, 0]  # Vmag
    node_pred_full[:, 1] = theta_reconstructed                # θ from Δθ
    node_pred_full[:, 2] = node_pred_partial[node_slice, 1]  # P
    node_pred_full[:, 3] = node_pred_partial[node_slice, 2]  # Q
    
    # Existing physics loss computation uses node_pred_full[N_i, 4] — unchanged
```

**Note**: `delta_theta_offsets` parallels the existing `node_slice` pattern used for per-graph node indexing. The forward() output `delta_theta_pred` is a single concatenated `[ΣE_fwd_i]` tensor because `forward_edge_mask` is applied inside forward() on the batched edge index.

**Loss combination:**

```python
# Updated formula:
if model.angle_mode == "edge_delta":
    mse_total = mse_node + loss_delta_theta  # node MSE (Vmag,P,Q) + edge MSE (Δθ)
else:
    mse_total = mse  # existing 4-col node MSE

loss = mse_total + eff_w_phys * physics + eff_w_ptdf * ptdf_loss
```

**Note**: `mse_node` here is the masked MSE on [Vmag, P, Q] only (3 cols). The angle component of MSE is entirely replaced by `loss_delta_theta` on edges. No additional weighting hyperparameter — both are MSE losses on per-unit quantities.

### 7. Update `_masked_mse_loss` for 3-Column Node Prediction

When `angle_mode="edge_delta"`, node prediction is [N, 3] = [Vmag, P, Q]. The masked MSE penalizes only unknowns:

| Bus type | Unknown quantities | Penalized cols in [Vmag, P, Q] |
|----------|-------------------|-------------------------------|
| PQ | Vmag, Vang | col 0 (Vmag) — angle handled by Δθ MSE |
| PV | Vang, Q | col 2 (Q) — angle handled by Δθ MSE |
| Slack | P, Q | col 1 (P), col 2 (Q) |

```python
def _masked_mse_loss(node_pred, y, batch, angle_mode="node"):
    if angle_mode == "edge_delta":
        # node_pred: [N, 3] = [Vmag, P, Q]
        # y: [N, 4] = [Vmag, Vang, P, Q] — extract cols 0, 2, 3
        y_mapped = torch.stack([y[:, 0], y[:, 2], y[:, 3]], dim=-1)  # [N, 3]
        
        mask = torch.zeros_like(node_pred, dtype=torch.bool)
        mask[batch.pq_mask, 0] = True    # Vmag unknown at PQ
        mask[batch.pv_mask, 2] = True    # Q unknown at PV
        mask[batch.slack_mask, 1] = True # P unknown at Slack
        mask[batch.slack_mask, 2] = True # Q unknown at Slack
        
        diff = (node_pred - y_mapped) ** 2
        return diff[mask].mean()
    else:
        # existing 4-col logic unchanged
        ...
```

### 8. Update `evaluate_gnn_on_test_set`

Add per-graph θ reconstruction and assembly of [N,4] output for backward compatibility:

```python
def evaluate_gnn_on_test_set(model, ..., angle_mode=None):
    ...
    angle_mode = angle_mode or getattr(model, 'angle_mode', 'node')
    
    theta_recon_times = []
    
    for data in test_dataset:
        data = data.to(device)
        
        with torch.no_grad():
            if angle_mode == "edge_delta":
                node_pred_partial, _, _, delta_theta_pred = model(data, return_embeddings=True)
                
                # Time θ reconstruction separately
                t0 = time.perf_counter()
                theta = reconstruct_theta_from_delta(
                    delta_theta_pred,
                    data.bfs_edge_idx, data.bfs_signs,
                    data.bfs_node_order, data.bfs_parent,
                    data.x.size(0),
                )
                theta_recon_times.append((time.perf_counter() - t0) * 1000)
                
                # Assemble [N, 4] = [Vmag, θ, P, Q]
                node_pred = torch.zeros(data.x.size(0), 4, device=device)
                node_pred[:, 0] = node_pred_partial[:, 0]  # Vmag
                node_pred[:, 1] = theta                     # reconstructed θ
                node_pred[:, 2] = node_pred_partial[:, 1]  # P
                node_pred[:, 3] = node_pred_partial[:, 2]  # Q
            else:
                node_pred, _, _, _ = model(data, return_embeddings=True)
        
        # ... existing override of known variables + metrics (unchanged) ...
    
    if theta_recon_times:
        metrics["theta_reconstruction_time_ms"] = np.mean(theta_recon_times)
```

### 9. Thread `angle_mode` Through Sweep and run_info

**`train_power_flow_gnn`**: Add `angle_mode="node"` parameter. Pass to `PowerFlowGNN(...)` and `PowerFlowDataset(...)`.

**`run_hparam_sweep`**: Add `angle_mode` to sweep grid or fixed params. Thread through to `train_power_flow_gnn`.

**`make_run_key`**: Include `angle_mode` in key string (only if not "node" to keep backward-compatible keys).

**`run_info` dict**: Add `"angle_mode": angle_mode` to saved metadata.

```python
# In make_run_key:
if angle_mode == "edge_delta":
    parts.append("edgeΔθ")

# In run_info:
run_info["angle_mode"] = angle_mode
```

### 10. Fixed-Length Return Tuple — Always Include `delta_theta_pred`

**Always return a fixed-size tuple** with `delta_theta_pred=None` when `angle_mode="node"`. This avoids fragile if/else unpacking at every callsite.

```python
# forward() always returns:
#   Without return_embeddings:
#     (node_pred, ptdf_pred, delta_theta_pred)  — 3-tuple
#   With return_embeddings=True:
#     (node_pred, h_nodes, h_edges, delta_theta_pred)  — 4-tuple
#
# When angle_mode="node": delta_theta_pred is None
# When angle_mode="edge_delta": delta_theta_pred is [E_fwd] tensor
```

Update the return statements in `forward()`:

```python
    if return_embeddings:
        return node_pred, h_nodes, h_edges, delta_theta_pred
    
    H_W = h_edges @ self.ptdf_W
    ptdf_pred = H_W @ h_nodes.T
    return node_pred, ptdf_pred, delta_theta_pred
```

All callsites unpack uniformly:

**Before:**
```python
node_pred, h_nodes, h_edges = model(batch, return_embeddings=True)
```

**After:**
```python
node_pred, h_nodes, h_edges, delta_theta_pred = model(batch, return_embeddings=True)
```

**Non-training callsites** (e.g. `batched_gnn_forward`):
```python
node_pred, ptdf_pred, _ = model(batch)  # discard delta_theta_pred
```

## Acceptance Criteria

- [ ] `PowerFlowGNN(angle_mode="edge_delta")` produces node output `[N, 3]` and edge angle output `[E_fwd]`
- [ ] `PowerFlowGNN(angle_mode="node")` behavior is completely unchanged (default)
- [ ] Both `head_mode="standard"` and `head_mode="with_encoder"` work with `angle_mode="edge_delta"`
- [ ] BFS precomputation produces correct spanning tree (verify: reconstructed θ from true Δθ matches true θ within float precision)
- [ ] `y_delta_theta` target matches ground truth `θ_from − θ_to` for all forward edges
- [ ] Training runs without error with batch_size > 1 and `angle_mode="edge_delta"`
- [ ] Physics loss receives correct reconstructed θ (gradients flow through reconstruction)
- [ ] `torch.autograd.gradcheck` passes on `reconstruct_theta_from_delta` with a small (N=5) graph — confirms gradient correctness through the sequential BFS loop
- [ ] `evaluate_gnn_on_test_set` returns standard `[N, 4]` output regardless of angle_mode
- [ ] `theta_reconstruction_time_ms` is reported in evaluation metrics
- [ ] Δθ MSE loss decreases during training (confirms the head is learning)
- [ ] `angle_mode` appears in `run_info` and `make_run_key`
- [ ] `forward()` returns fixed-length tuples (3-tuple or 4-tuple with return_embeddings) for both angle modes
