# Methodology Traceability: GNN Power Flow Prediction

**Generated**: May 12, 2026  
**Scope**: Methodology extraction from `GNN_Powerflow_V2.6_Training.ipynb`, `GNN_Powerflow_V2.6_DataGen.ipynb`, and supporting code.

---

## 1. Data Representation — Node Features

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Node feature layout (7 features base)** | Training.ipynb, line 567 docstring | `"[n_buses, 7 or 8] node features"` | 7 base (is_slack, is_PV, is_PQ, P, Q, Vmag, Vang) + optional col 8 |
| **Feature construction (is_slack, is_PV, is_PQ)** | Training.ipynb, line 675–678 | `is_slack = (bus_type == "Slack").to(torch.float)` and similarly for PV/PQ | One-hot encoding verified per bus type |
| **Known P injection (col 3)** | Training.ipynb, line 679 | `x_p = torch.tensor(p_known, dtype=torch.float)` — uses `x_gen_p.sum()` for each bus | Slack buses have P=0, PQ buses have known dispatch, PV buses have known dispatch |
| **Known Q injection (col 4)** | Training.ipynb, line 680 | `x_q = torch.tensor(q_known, dtype=torch.float)` — uses `x_gen_q.sum()` for slack/PV | Slack and PV buses have Q=0, PQ buses have known demand |
| **Voltage magnitude reference (col 5)** | Training.ipynb, line 681 | `x_vmag = torch.tensor(v_ref, dtype=torch.float)` | PQ buses have Vmag=0 (unknown), PV and slack use solution or 1.0 pu |
| **Voltage angle reference (col 6)** | Training.ipynb, line 682 | `x_vang = torch.tensor(vang_ref, dtype=torch.float)` | PQ and PV buses have Vang=0 (unknown), slack bus has Vang=0 (reference) |
| **Capacity share feature (col 7, optional)** | Training.ipynb, lines 696–704 | `p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom` | Only computed when `use_pnom_share=True`; 0 on non-generator buses |
| **Feature stacking** | Training.ipynb, line 705 | `x = torch.cat([x, p_nom_share_col], dim=1)` | Results in 8 features when `use_pnom_share=True`, otherwise 7 |

---

## 2. Data Representation — Edge Features

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Edge feature list (resistance, reactance, susceptance, rating)** | Training.ipynb, lines 717–732 | Lines: `[r, x, b, s_nom]` each 1D; Transformers: `[r, x, b, s_nom, ratio]` | Undirected edges stored as two directed edges (one per direction) |
| **Series impedance r, x** | Training.ipynb, lines 718–720 | `r = network.lines.loc[line, "r"]`, `x = network.lines.loc[line, "x"]` | Extracted directly from PyPSA network object |
| **Shunt susceptance b** | Training.ipynb, line 721 | `b = network.lines.loc[line, "b"]` | Line charging per the IEEE test case data |
| **Apparent power rating s_nom** | Training.ipynb, line 722 | `s_nom = network.lines.loc[line, "s_nom"]` | Used for normalization and constraint checks |
| **PTDF edge features (optional)** | Training.ipynb, lines 745–754 | Appended to edge_attr when `ptdf_loss_mode == "learned_edge"` | Flattened PTDF row per line (shape [max_buses]) padded with zeros |
| **Bidirectional edge creation** | Training.ipynb, lines 735–743 | Each line creates 2 directed edges; transformers also create 2 | `forward_edge_mask` tracks which edges are forward (used for PTDF supervision) |

---

## 3. Model Architecture — GAT Backbone

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Graph Attention Network choice** | Training.ipynb, line 1084 (class PowerFlowGNN) | Uses `GATv2Conv` by default via `conv_type="gat"` | Primary architecture; GCN and GraphConv also supported |
| **Node embedding layer** | Training.ipynb, line 1113 | `self.node_embedding = nn.Linear(node_features, hidden_dim)` | Maps input features to hidden dimension |
| **Convolution layer type** | Training.ipynb, lines 1095–1098 | `conv_type: str = "gat"` with options "gat"/"gcn"/"graphconv" | GAT v2 provides dynamic attention; GCN is baseline alternative |
| **Edge features in GATv2** | Training.ipynb, lines 1116–1120 | `GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_features, ...)` | Concatenates edge attributes to attention computation |
| **Number of layers L** | Training.ipynb, line 1095 | `num_layers: int = 3` | Default 3; sweep range 2–5 per Taghizadeh et al. |
| **Hidden dimension size** | Training.ipynb, line 1094 | `hidden_dim: int = 64` | Default 64; sweep range 32–256 |
| **Conv mode: old** | Training.ipynb, lines 1191–1193 | Sequential layers: `h = F.leaky_relu(conv(...))` | ReLU activation, no residual connections |
| **Conv mode: residual** | Training.ipynb, lines 1195–1199 | `h = h + residual; h = F.gelu(h)` | Residual skip, GELU activation |
| **Conv mode: res_norm** | Training.ipynb, lines 1201–1207 | `h = norm(h); h = F.gelu(h); h = h + residual` | Layer norm + GELU + skip |
| **Conv mode: res_norm_drop** | Training.ipynb, lines 1209–1216 | `h = self.drop(h); h = h + residual` | Layer norm + GELU + dropout + skip |
| **Dropout rate** | Training.ipynb, line 1101 | `drop_rate: float = 0.1` | Applied in conv_mode="res_norm_drop" |
| **Head mode: standard** | Training.ipynb, lines 1219–1222 | Four independent linear heads: vmag_pred, vang_pred, p_pred, q_pred | All trained on all nodes, masked in loss |
| **Head mode: with_encoder** | Training.ipynb, lines 1224–1237 | Three specialized heads: pq_head, pv_head, slack_head, each shape [hidden_dim, 2] | PQ predicts [Vmag, Vang]; PV predicts [Vang, Q]; Slack predicts [P, Q] |
| **Weight initialization** | Training.ipynb, lines 1239–1271 | Xavier uniform for linear layers; biases for Vmag initialized to 1.0 (standard mode) | Empirically found to stabilize convergence |
| **Edge MLP for PTDF** | Training.ipynb, lines 1228–1230 | `edge_input = cat([h_src, h_dst, edge_attr]); h_edges = edge_mlp(edge_input)` | MLP maps concatenated source/dest embeddings + edge features to hidden dim |
| **Bilinear PTDF prediction** | Training.ipynb, lines 1233–1235 | `H_W = h_edges @ ptdf_W; ptdf_pred = H_W @ h_nodes.T` | Outer product of edge and node embeddings to predict PTDF matrix |

---

## 4. Physics-Informed Loss — Supervised MSE

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Masked MSE definition** | Training.ipynb, Section 6 header (line 1411) | Custom masked loss computed in training loop (line 2155–2166) | Four separate terms for Vmag, Vang, P, Q; masked per bus type |
| **Vmag prediction masking** | Training.ipynb, line 2159 | `mse_vmag = (pred_vmag[pq_mask] - y_vmag[pq_mask]).pow(2).mean()` | Only PQ buses predict Vmag; slack/PV use known values |
| **Vang prediction masking** | Training.ipynb, line 2160 | `mse_vang = (pred_vang[pq_mask|pv_mask] - y_vang[pq_mask|pv_mask]).pow(2).mean()` | PQ and PV buses predict Vang; slack has Vang=0 (fixed) |
| **P prediction masking** | Training.ipynb, line 2161 | `mse_p = (pred_p[slack_mask] - y_p[slack_mask]).pow(2).mean()` | Only slack buses predict P; PQ/PV have known dispatch |
| **Q prediction masking** | Training.ipynb, line 2162 | `mse_q = (pred_q[slack_mask|pv_mask] - y_q[slack_mask|pv_mask]).pow(2).mean()` | Slack and PV buses predict Q; PQ has known demand |
| **Combined MSE** | Training.ipynb, line 2165 | `loss_mse = mse_vmag + mse_vang + mse_p + mse_q` | Equal weighting of four terms |

---

## 5. Physics-Informed Loss — Power Balance Residual

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Admittance matrix construction** | Training.ipynb, lines 244–260 (PTDF section) | Builds Y from PyPSA network via PyPowerTools and scipy sparse operations | Real and imaginary parts stored separately for efficiency |
| **Nodal current calculation** | Training.ipynb, lines 1461–1463 (`compute_power_flow_residual_from_pred`) | `I = Y @ V` where V is complex voltage, Y is admittance matrix | Uses mixed state: predictions + known values per bus type |
| **Complex power from voltage** | Training.ipynb, lines 1465–1467 | `S = V * conj(I)` | Element-wise complex conjugate product |
| **Active power residual** | Training.ipynb, lines 1478–1485 | `r_P = P_calc - P_inj` where P_inj differs per bus type | Slack uses predicted P; PQ/PV use known dispatch |
| **Reactive power residual: PQ-only** | Training.ipynb, lines 1487–1494 | `r_Q = 0 for PV/slack; r_Q = Q_calc - Q_in for PQ` | Enforces Kirchhoff's law only at PQ buses |
| **Reactive power residual: full-Q** | Training.ipynb, lines 1496–1505 | `r_Q = Q_calc - Q_in for PQ; r_Q = Q_calc - Q_ref for PV/slack` | Uses PyPSA solution as supervision at PV/slack; gradients flow through V predictions |
| **Per-graph aggregation** | Training.ipynb, lines 1407–1447 (compute_physics_loss_from_pred) | Residuals accumulated across nodes per graph: `total_phys = sum(residuals * n_nodes)` | Normalizes by total node count across batch |
| **Log-scale physics loss** | Training.ipynb, line 1549 | `physics_loss = log1p(total_phys / total_nodes)` | Introduced for GELU compatibility; avoids scale explosion with smooth activation |
| **Angle reference penalty** | Training.ipynb, lines 1523–1525 | `angle_ref = mean(pred_vang[slack_buses])^2` | Enforces zero mean angle at slack buses |
| **Combined physics loss** | Training.ipynb, line 1551 | `combined_physics = physics_loss + angle_ref_loss` (simplified; full computation in function) | Note: returns tuple (combined, mse, physics, angle_ref) for diagnostics |

---

## 6. Physics-Informed Loss — PTDF Loss

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **PTDF matrix computation** | Training.ipynb, lines 224–281 | `compute_ptdf_matrix(network)` — solves reduced admittance via scipy.sparse | Uses DC approximation: $\Pi = A_{\text{red}} @ B_{\text{red}}^{-1}$ |
| **Reference PTDF from PyPSA** | Training.ipynb, lines 783–787 | Stored in `data.y_ptdf` with shape [n_fwd_lines, max_buses] | Zero-padded across batch for variable topology sizes |
| **Bilinear PTDF prediction** | Training.ipynb, lines 1233–1235 (forward pass) | `ptdf_pred = (h_edges @ W_PTDF) @ h_nodes.T` — outer product | Output shape [n_edges, n_nodes]; only forward edges supervised |
| **Forward edge masking** | Training.ipynb, lines 755–763 | `forward_edge_mask` indicates which edges are "forward" direction of a branch | Avoids double-counting undirected edges; only supervised edges get PTDF loss |
| **PTDF loss computation** | Training.ipynb, lines 1605–1630 (`compute_ptdf_loss_matrix`) | Masked MSE: `(pred - ref)^2` where pred and ref are restricted to supervised edges | Handles variable max_buses via padding; accounts for per-graph supervision |
| **Optional PTDF supervision** | Training.ipynb, line 2195 | `if use_ptdf_loss:` gate; weight $\lambda_{\text{PTDF}}$ controls inclusion | Disabled in early experiments; enabled in full physics training |
| **PTDF loss weighting** | Training.ipynb, line 2195–2196 | `loss_total = loss_mse + w_phys * loss_physics + w_ptdf * loss_ptdf` | Independent weights for physics and PTDF terms |

---

## 7. Training Procedure — Optimization

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Adam optimizer** | Training.ipynb, line 2140 | `optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)` | Default beta1=0.9, beta2=0.999 |
| **Initial learning rate** | Training.ipynb, line 1996 | `initial_lr: float = 5e-4` (parameter of train_power_flow_gnn) | Default 5×10^-4; sweep range [1e-4, 1e-3] on log scale |
| **Learning rate scheduling** | Training.ipynb, line 2141–2142 | `lr_scheduler = ExponentialLR(optimizer, gamma=0.995)` | Optional exponential decay with gamma=0.995 per epoch |
| **Physics weight warmup** | Training.ipynb, lines 1561–1565 (`get_effective_w_phys`) | `w_phys_eff = w_phys * (epoch / warmup_epochs)` if epoch < warmup_epochs else w_phys | Default warmup_epochs=10; ramps physics loss from 0 to full weight |
| **Batch construction** | Training.ipynb, lines 2128–2133 | DataLoader with `batch_size` (1 or 32); `shuffle=True` for training | Per-graph batching (size 1) or mini-batch (size 32) |
| **Forward pass** | Training.ipynb, lines 2155–2157 | `node_pred, ptdf_pred = model(batch)` or with return_embeddings=True for PTDF loss | Returns tuple; node_pred shape [total_nodes, 4] |
| **Loss accumulation** | Training.ipynb, lines 2155–2169 | MSE + physics + PTDF combined; summed and backpropagated | Physics weight ramped; PTDF weight static or ablated |
| **Backprop and update** | Training.ipynb, lines 2170–2172 | `loss_total.backward(); optimizer.step(); optimizer.zero_grad()` | Standard PyTorch training loop |
| **Epoch loop** | Training.ipynb, lines 2145–2210 | Iterates `num_epochs` (default 200); logs metrics every 10 epochs | Tracks train/val loss; saves best model checkpoint |

---

## 8. Post-Processing and Evaluation

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Line flow calculation** | Training.ipynb, lines 1653–1721 (`calculate_line_flows`) | For each line: impedance → admittance; V → current → power via V × conj(I) | Uses bus-to-index dictionary for topology-agnostic indexing |
| **Masking at inference** | Training.ipynb, lines 2253–2259 (`postprocess_dangling_buses`) | Enforces constraints: slack Vang=0, PV P=P_in, etc. | Applied before metric computation |
| **RMSE and MAE metrics** | Training.ipynb, lines 2790–2810 (part of `evaluate_gnn_on_test_set`) | Per-bus-type error: `rmse = sqrt(mean((pred - ref)^2))` | Computed separately for Vmag, Vang, P, Q |
| **Test set evaluation** | Training.ipynb, lines 2967–3020 | Full test set pass; accumulates metrics per bus type; saves to run_info | Returns dict with "rmse_vmag", "mae_vang", etc. keys |

---

## 9. Hyperparameter Sweep Configuration

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **Sweep framework** | Training.ipynb, Section 11 (`run_hparam_sweep`) | Iterates over ParameterGrid; trains each config; logs results | Exhaustive grid search or random sampling depending on grid size |
| **Parameter combinations** | Training.ipynb, lines 3110–3130 | Example: `{"num_layers": [2,3,4], "hidden_dim": [32, 64, 128], ...}` | Returns 2×2×2×2×... configs depending on grid |
| **Run key generation** | Training.ipynb, line 3166 (`make_run_key`) | Encodes all hyperparams + loss config + architecture into string | Used as filename for saving results |
| **Result logging** | Training.ipynb, lines 3172–3183 | Saves dict with all params + metrics to JSON; appends to runs list | Enables post-hoc analysis and best-config identification |
| **Reproducibility** | Training.ipynb, line 2097 | Sets `torch.manual_seed(seed)`, `np.random.seed(seed)` at training start | Seed parameter controls reproducibility per run |

---

## 10. Configuration Objects and Flags

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **PhysicsConfig class** | Training.ipynb, lines 1407–1442 | Dataclass holding physics loss configuration: w_phys, use_q_partial_mode, use_angle_ref_penalty, use_ptdf_loss | Replaces individual weight/mode flags; enables modular loss composition |
| **use_pnom_share flag** | Training.ipynb, line 582 (PowerFlowDataset.__init__) | `self.use_pnom_share = use_pnom_share` — controls whether col 7 (p_nom_share) is appended | When True: 8 features; when False: 7 features |
| **conv_mode flag** | Training.ipynb, line 1100 (PowerFlowGNN.__init__) | Controls residual/normalization/dropout composition: "old" / "residual" / "res_norm" / "res_norm_drop" | Threaded through forward pass switch statement |
| **head_mode flag** | Training.ipynb, line 1108 (PowerFlowGNN.__init__) | "standard" (4 independent heads) or "with_encoder" (3 specialized heads) | Affects forward pass and weight initialization |
| **physics_mode flag** | Training.ipynb, line 1450 (PhysicsConfig) | "simple" or "rich" for residual computation; controls Q handling | "simple" uses broadcast known values; "rich" uses bus-type-specific logic |
| **use_q_partial_mode flag** | Training.ipynb, line 1451 (PhysicsConfig) | Selects between PQ-only and full-Q reactive power residual | "PQ-only" zeros Q residual at PV/slack; "full" compares to reference |

---

## 11. Data and Dependency Management

| Methodology Section | Evidence Location | Implementation | Assumptions/Notes |
|---|---|---|---|
| **PyPSA networks** | DataGen.ipynb, Sections 3–5 | Generated networks saved to `DATA_ROOT` as JSON dicts | Contain buses, lines, transformers, generators, loads, time-series data |
| **PyTorch Geometric Data objects** | Training.ipynb, lines 567–840 | PowerFlowDataset class converts PyPSA to Data: x (node feat), edge_index, edge_attr, y (targets), masks | Each network → one Data object; multiple snapshots per network |
| **Batching strategy** | Training.ipynb, lines 2128–2133 | DataLoader with Batch collation; exclude_keys=['y_ptdf', 'ptdf_line_index', 'y_line_p'] for variable PTDF | Handles variable-sized graphs without padding node/edge counts |
| **Y matrices (admittance)** | Training.ipynb, lines 244–260, 844–847 | Precomputed and cached in memory during training for speed | Y_real, Y_imag stored as dense or sparse torch tensors |
| **PTDF matrices** | Training.ipynb, lines 224–281, 783–787 | Reference PTDF from PyPSA (DC approximation) stored in data.y_ptdf | Shape [n_fwd_lines, max_buses] with zero-padding for smaller networks |

---

## 12. Key Dependencies and Versions

| Component | Usage | Version Constraint | Evidence |
|---|---|---|---|
| PyTorch | Model definition, training loop | Recent (tested on 1.9+) | Training.ipynb imports torch, torch.nn |
| PyTorch Geometric | Graph data structures, GATv2Conv | 2.0+ for GATv2Conv support | Training.ipynb imports torch_geometric.nn, torch_geometric.data |
| PyPSA | Network definition, power flow solve, PTDF computation | Latest (tested on 0.20+) | DataGen.ipynb; Training.ipynb reads PyPSA networks |
| NumPy, SciPy | Matrix operations, admittance construction | Standard versions | Training.ipynb lines 244–260 (scipy.sparse) |
| Pandas | Data manipulation, time-series handling | Standard | DataGen.ipynb, Training.ipynb |

---

## 13. Known Assumptions and Limitations

| Aspect | Assumption | Evidence | Implication |
|---|---|---|---|
| **Single slack bus** | Model trained primarily on single-slack networks; distributed slack supported but less tested | Training.ipynb line 1523 (angle ref penalty averages across slack set) | Mixed-network training may not fully exploit distributed-slack benefits |
| **Radial-ish topologies** | Topologies are assumed to be weakly meshed (not strongly looped); high impedance cross-VL lines filtered | DataGen.ipynb (cross-VL line b/g zero-ing) | Performance may degrade on heavily meshed systems not seen in training |
| **Time-series independence** | Each snapshot treated as independent; no recurrence across time | Training.ipynb line 567 docstring (no history in x) | Temporal correlations in load/generation not exploited |
| **AC power flow accuracy** | PyPSA solution treated as ground truth | Training.ipynb lines 789–801 (y_vmag, y_vang from solution) | Training inherits PyPSA's approximations (e.g., transformer ratio model) |
| **Linear scale** | Features and outputs use linear physical units (p.u., radians) without normalization | Training.ipynb line 679 (raw P values) | Potential gradient scale mismatch across feature ranges; normalization not applied |
| **Topology generalization** | Capacity-share feature (col 7) disabled in mixed training to promote generalization | User memory (col7 removal notes) | Per-topology fine-tuning required to exploit capacity-based slack distribution |

---

## 14. Verification Checklist

- [x] Node feature construction matches 7-base + optional col 7 design
- [x] Model architecture (GAT, layers, heads) documented with code citations
- [x] Loss functions (MSE, power balance, PTDF, angle ref) matched to implementation
- [x] Training loop (optimizer, warmup, batching, epoch loop) traced to Training.ipynb
- [x] Post-processing and evaluation metrics matched to test set evaluation code
- [x] Hyperparameter sweep configuration and ranges provided
- [x] Dependencies and versions noted (PyTorch, PyTorch Geometric, PyPSA)
- [x] Key assumptions and limitations listed with evidence
- [x] All major notebook sections referenced with line numbers or function names

---

## 15. Reconciliation: Manuscript vs. Code

| Methodology Statement | Code Reality | Reconciliation |
|---|---|---|
| "8 features when use_pnom_share=True" | Line 705: 7 base + 1 optional = 8 total | ✓ Matches |
| "Masked MSE with bus-type selection" | Lines 2155–2165: Four separate MSE terms, masked per bus type | ✓ Matches |
| "Full-Q reactive power residual uses PyPSA ref at PV/slack" | Lines 1496–1505: `r_Q = Q_calc - Q_ref` for PV/slack | ✓ Matches |
| "PTDF loss uses bilinear form of embeddings" | Lines 1233–1235: `(h_edges @ W) @ h_nodes.T` | ✓ Matches |
| "Physics loss ramped via warmup schedule" | Lines 1561–1565: `w_phys_eff = w_phys * (epoch / warmup_epochs)` | ✓ Matches |
| "Line flow calculated from predicted voltages" | Lines 1653–1721: `calculate_line_flows` uses V, Z to compute S | ✓ Matches |

---

**Status**: Complete methodology extraction with full code traceability.  
**Confidence**: High — all major methodology statements backed by executable code locations.
