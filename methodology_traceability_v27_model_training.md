# Methodology Traceability — Model Architecture & Training (V2.7)

**Generated:** 2026-07-03
**Source:** `GNN_Powerflow_V2.7_Training.ipynb` (45 code cells).
**Companion manuscript text:** `methodology_model_training.md`.
**Scope:** Data preparation (graph construction), model architecture, physics-informed loss, training, evaluation, sweep. Dataset *generation* is out of scope (separate section).

Prefer executable-path evidence over comments. Cell numbers are 0-based code-cell indices from the extraction.

---

## 1. Data Preparation — Graph Construction

| Step ID | Method step (paper wording) | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| DP-1 | Snapshot → graph sample | Cell 12, `PowerFlowDataset` / `_create_graph_data` | PyTorch Geometric `Data` | One graph per (network, snapshot) |
| DP-2 | Node feature vector (7 base) | Cell 12: `x = torch.stack([is_slack, is_pv, is_pq, x_p, x_q, x_vmag, x_vang], dim=1)` | — | Order fixed; indices 0–6 |
| DP-3 | Bus-type zero-masking of unknowns | Cell 12: P set for PQ/PV, Q for PQ, Vmag for PV/slack, Vang for slack | — | Unknowns are 0, not NaN |
| DP-4 | Optional capacity-share feature (8th) | Cell 12: `if self.use_pnom_share: x = torch.cat([x, p_nom_share_col], dim=1)` | `use_pnom_share` flag | Static; 0 on non-generator buses |
| DP-5 | Edge features `[r, x, b_half, tau, g_ser, b_ser]` | Cell 12: `attr = [r, x, b_half, tap, g_ser, b_ser]`, `g_ser,b_ser = 1/(r+jx)` | flat-pu convention | 6 entries; `tau=1` for lines |
| DP-6 | Bidirectional interleaved edges | Cell 12: forward/reverse `[fwd0,rev0,fwd1,rev1,...]` | — | Forward edges at even indices |
| DP-7 | `forward_edge_mask` (even idx) | Cell 12: bool mask over 2E edges | `ptdf_branch_mode` (lines vs all) | Selects supervised forward edges |
| DP-8 | `dc_flow_mask` (forward LINE edges only) | Cell 12: excludes transformers | — | DC flow ill-conditioned for trafos |
| DP-9 | Node target `y=[V*,θ*,P*,Q*]` | Cell 12: `y = torch.stack([v_mag, v_ang, p_bus, q_bus], dim=1)` | PyPSA solved state | Shape `[N,4]` |
| DP-10 | Auxiliary targets: `y_ptdf`, `y_line_p`, `y_line_q`, `ptdf_line_index` | Cell 12; removed before batch in `collate_with_ptdf` | — | Variable shape → excluded from collation |
| DP-11 | `y_delta_theta` + BFS metadata | Cell 12: `precompute_bfs_order(...)`; keys `bfs_edge_idx, bfs_signs, bfs_node_order, bfs_parent, y_delta_theta` | `angle_mode in {edge_delta, both}` | Used by `reconstruct_theta_from_delta` |
| DP-12 | No feature standardization | Cell 12: per-unit values used as-is | — | Only zero-masking applied |

---

## 2. Model Architecture — `PowerFlowGNN` (Cell 13)

| Step ID | Method step | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| MA-1 | Backbone default GATv2, alternatives | Cell 13: `conv_type="gatv2"` → `GATv2Conv(..., edge_dim=6, heads=4, concat=True)`; `transformer`/`gcn`/`graphconv` | `conv_type`, `heads=4` | GCN/GraphConv ignore edge features |
| MA-2 | Node embedding | Cell 13: `self.node_embedding = nn.Linear(node_features, hidden_dim)` | `hidden_dim=64` | — |
| MA-3 | Unified conv block | Cell 13: `h_in; h=conv(...); norm; act; drop; if use_residual: h=h+h_in` | `num_layers=3`, `norm_type`, `activation="leaky_relu"`, `use_dropout`, `use_residual` | Order fixed |
| MA-4 | Activation options | Cell 13: leaky_relu default; relu/gelu/elu/silu | `activation` | — |
| MA-5 | Optional global pooling | Cell 13: `h_global=global_proj(cat([mean,max])); h_nodes = h_nodes + h_global[batch_idx]` | `use_global_pool` | Additive per-node residual |
| MA-6 | angle_mode = node | Cell 13: node head → `[N,4]` `[V,θ,P,Q]` | `angle_mode="node"` | Direct angle head |
| MA-7 | angle_mode = edge_delta | Cell 13: `edge_angle_pred=nn.Linear(hidden,1)`; node head `[N,3]`; θ via `reconstruct_theta_from_delta` | `angle_mode="edge_delta"` | Edge head zero-init; BFS integrate |
| MA-8 | angle_mode = both | Cell 13: node θ + edge Δθ | `angle_mode="both"` | Jointly supervised |
| MA-9 | vmag_mode absolute/residual | Cell 13: absolute bias init 1.0; residual predicts δV around 1.0 (zero-init) | `vmag_mode` | — |
| MA-10 | head_mode standard (4 heads) | Cell 13: `vmag_pred/vang_pred/p_pred/q_pred` | `head_mode="standard"` | Masked per bus type in loss |
| MA-11 | head_mode with_encoder | Cell 13: `pq_head/pv_head/slack_head` (PQ:[V,θ], PV:[θ,Q], slack:[P,Q]) | `head_mode="with_encoder"` | Structural known/unknown split |
| MA-12 | Edge MLP | Cell 13: `edge_mlp=Sequential(Linear(2h+edge_feat,h),LeakyReLU())` | — | Input `[h_src,h_dst,edge_attr]` |
| MA-13 | Bilinear PTDF head | Cell 13: `H_W=h_edges@ptdf_W; ptdf_pred=H_W@h_nodes.T` | `ptdf_W` param | Shape `[2E,N]` |
| MA-14 | Xavier init + special cases | Cell 13: xavier_uniform; Vmag bias 1.0 (absolute); edge-angle zero | — | Empirically stabilizes |
| MA-15 | Forward signature | Cell 13: returns `(node_pred, ptdf_pred, delta_theta_pred)` or `(node_pred, h_nodes, h_edges, delta_theta_pred)` | `return_embeddings` | — |

---

## 3. Physics-Informed Loss (Cell 14)

| Step ID | Method step | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| PL-1 | `PhysicsConfig` dataclass | Cell 14: fields for weights, modes, fractions, activation | — | Central loss config |
| PL-2 | Masked MSE per bus type | Cell 16: `_masked_mse_loss(node_pred, batch.y, batch, angle_mode)` | `angle_mode` | PQ:[V,θ], PV:[θ,Q], slack:[P,Q] |
| PL-3 | edge_delta target mapping | Cell 16: `y_mapped=[V,P,Q]` + `loss_delta_theta=MSE(Δθ_pred, y_delta_theta)` | `angle_mode` | Node target reduced to 3 |
| PL-4 | Power-balance residual (Y-bus) | Cell 14 `compute_power_flow_residual_from_pred`: `I=Y@V; p_calc,q_calc; residual/b_diag` | `use_power_balance`, `w_P=0.5`, `w_Q=0.5` | Normalized by diagonal susceptance |
| PL-5 | Injection assembly by bus type | Cell 14: PQ both known; PV P known, Q pred; slack both pred | — | — |
| PL-6 | PQ-only vs full-Q | Cell 14: `use_q_partial_mode` excludes PV from Q residual | `use_q_partial_mode` | Default PQ-inclusive |
| PL-7 | Edge-local O(E) physics | Cell 14: `scatter(p_edge, src)`, `p_calc=V²g_diag+p_off` | `angle_mode in {edge_delta, both}` | Avoids dense Y-bus |
| PL-8 | Angle reference penalty | Cell 14/16: `angle_ref = slack_angle_mean()**2` | `use_angle_ref_penalty` | Soft constraint |
| PL-9 | Flow losses (4-way DC/AC × local/global) | Cell 14: DC `f=Δθ/x`; AC full π P/Q | `fraction_dcf_local/global`, `fraction_acf_local/global`, `flow_target` | local=Δθ head, global=node θ diff |
| PL-10 | PTDF loss modes | Cell 14: `matrix`/`flows`/`mixed`/`learned_edge`/`stepwise` | `ptdf_loss_mode`, `ptdf_alpha`, `ptdf_branch_mode` | learned_edge adds params |
| PL-11 | Adaptive weighting | Cell 16: `eff_w = min(fraction*mse/max(subloss,1e-8), max_w)` | `loss_weight_mode="adaptive"` | Keeps terms commensurate with MSE |
| PL-12 | Activation schedule (ramp) | Cell 16: `eff_w *= activation_scale(epoch, start, stop, ramp_epochs)` | `*_activation_start`, `activation_ramp_epochs=5` | Linear ramp on/off |
| PL-13 | Total loss | Cell 16: `loss = mse + loss_delta_theta + eff_w_phys*physics + eff_w_ptdf*ptdf + flow_loss_val` | — | Sum of all active terms |

---

## 4. Training Procedure — `train_power_flow_gnn` (Cell 16)

| Step ID | Method step | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| TR-1 | Data split 70/15/15 | Cell 16: shuffle → train 70% / val 15% / test 15% | `seed=42` | Per-network split |
| TR-2 | Optimizer Adam lr=1e-3 | Cell 16: `Adam(model.parameters(), lr=lr)`, default `lr=0.001` | `lr` | betas (0.9, 0.999) |
| TR-3 | LR scheduler | Cell 16: `ReduceLROnPlateau(mode="min", factor=0.5, patience=10)` | — | Steps on val loss |
| TR-4 | Learnable PTDF param group | Cell 16: `optimizer.add_param_group({'params': ptdf_params})` | `ptdf_loss_mode="learned_edge"` | Only when learned PTDF used |
| TR-5 | Epochs / batch defaults | Cell 16: `num_epochs=200`, `batch_size=1` | — | Sweep driver uses batch 32 |
| TR-6 | Train/val loop | Cell 16: forward `return_embeddings=True`; compute all losses; backward; step | — | — |
| TR-7 | Best-state retention | Cell 16: save state dict when `val_loss < best_val_loss` | — | — |
| TR-8 | run_info / history saved | Cell 16: per-epoch train/val traces of every loss + effective weights | — | See `history` keys in extraction |
| TR-9 | Xavier init | Cell 13 (applied at model build) | — | — |

---

## 5. Post-Processing & Evaluation (Cell 17)

| Step ID | Method step | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| EV-1 | AC line flows (π-model) | Cell 17 `calculate_line_flows`: `I_from=(V_from-V_to)*y_series + V_from*y_shunt` | — | From predicted voltages |
| EV-2 | Line flows from Δθ | Cell 17 `calculate_line_flows_from_delta_theta`: tap-adjusted π-model | `angle_mode edge_delta` | No θ reconstruction |
| EV-3 | Known-variable override at inference | Cell 17 `evaluate_gnn_on_test_set`: PV/slack Vmag, slack angle, PQ P overridden | — | Enforces known/unknown split |
| EV-4 | Metrics MAE/RMSE per bus type | Cell 17: vmag_mae, vang_mae, p_mae, q_mae, line_flow_mae | — | Aggregated over test set |
| EV-5 | Timing | Cell 17: solve_time_ms, postproc_time_ms, total_time_ms, batch_solve_time_ms | CPU | — |

---

## 6. Hyperparameter Sweep — `run_hparam_sweep` (Cell 20)

| Step ID | Method step | Implementation evidence | Dependencies / config | Assumptions / limitations |
|---|---|---|---|---|
| HS-1 | Cartesian product sweep | Cell 20: nested loops over all tuple args | — | Can be 1000s of runs |
| HS-2 | Swept classical hparams | Cell 20: `batch_sizes,(learning_rates),(hidden_dims),(num_layers_list),(activations),(drop_rates)` | — | Defaults noted in extraction |
| HS-3 | Swept architectural options | Cell 20: `conv_types=("gatv2","transformer")`, `angle_modes`, `vmag_modes`, `use_global_pools`, `head_modes`, `use_residuals`, `norm_types` | — | V2.7 additions |
| HS-4 | Swept loss schedules | Cell 20: `physics_activation_list`, `ptdf_activation_list`, `dcf/acf_*_activation_list`, `activation_ramp_epochs_list` | — | Per-term activation |
| HS-5 | Checkpoint / resume | Cell 20: `save_path` pickle, skip done via stable run-key hash | `save_path` | Enables interrupt/continue |

---

## 7. V2.7 vs V2.6 delta (for reviewer defensibility)

| Feature | V2.6 | V2.7 |
|---|---|---|
| angle_mode | node only | node / edge_delta / both |
| Physics compute | Y-bus | Y-bus OR O(E) edge-local |
| Flow loss | single weight | 4 sub-losses (DC/AC × local/global) |
| Weighting | fixed / global adaptive | per-sub-loss adaptive (fraction of MSE) |
| Activation | binary gate | linear ramp per component |
| vmag_mode | absolute | absolute / residual |
| Global pool | absent | optional context injection |
| Backbones | GATv2 | GATv2 / Transformer / GCN / GraphConv |
| Learned PTDF | absent | `learned_edge` mode |
| lr default | 5e-4 (old draft) | 1e-3 + ReduceLROnPlateau |

---

## 8. Unresolved ambiguities / notes

- `edge_features` constructor default is 4, but the dataset stores 6-entry edge vectors; the build path passes the actual count. Confirm no path uses the 4-default.
- `ptdf_loss_mode="stepwise"` and `ptdf_changepoint` exist but are experimental; omitted from manuscript text.
- Exact B-series sweep ranges (cells 21+) not fully enumerated here; Table ranges in the manuscript are representative and should be reconciled with reported experiments before submission.
- The old draft's explicit "full-Q" residual equation maps to `use_q_partial_mode=False`; confirm which mode produced the reported metrics.
