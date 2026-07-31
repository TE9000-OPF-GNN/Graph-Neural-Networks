# Hyperparameter Table — Traceability Record

**Generated:** 2026-07-25  
**Source runs:** `runs_combined_f5b` — 98 unique runs  
**Sweep files:** B1–B10a (`*_bn.json`), C1a–C1c (`sweep_C1*.json`)  
**Data directory:** `TRAINING_RESULTS_DIR` = `C:\Users\STSI\OneDrive - USN\Data_PF_GNN\training_results_saved\`

---

## 1. Scope

Table covers all hyperparameters stored as scalar keys in the pickled run dicts produced by `run_hparam_sweep` (V2.7 Training notebook).  
Keys classified as non-hyperparameter artefacts (model weights, history arrays, metrics, run_key, tag, etc.) are excluded.

Extraction method: custom `pickle.Unpickler` subclass bypassing `PowerFlowGNN` class dependency;  
unique value enumeration over 98 deduplicated runs (deduplication by `run_key`).

---

## 2. Sweep Inventory

| Sweep ID | File | n runs | Primary focus |
|----------|------|--------|---------------|
| B1 | `sweep_B1_dc_ac_local_both_bn.json` | 4 | `angle_mode`, `fraction_acf_local` |
| B2 | `sweep_B2_dc_ac_local_no_globals_bn.json` | 5 | `fraction_ptdf`, `fraction_dcf_local`, `fraction_acf_local` |
| B3 | `sweep_B3_dc_lg_ac_local_bn.json` | 4 | `fraction_ptdf`, `fraction_dcf_global` |
| B4 | `sweep_B4_full_curriculum_both_bn.json` | 4 | `angle_mode`, `fraction_acf_global` |
| B5 | `sweep_B5_dc_ac_local_ramp_bn.json` | 6 | `fraction_ptdf/dcL/acL`, `activation_ramp_epochs` |
| B6 | `sweep_B6_extended_dc_no_globals_bn.json` | 6 | all flow fractions, `activation_ramp_epochs` |
| B7 | `sweep_B7_full_curriculum_low_bn.json` | 6 | `fraction_physics`, `fraction_dcL/acL`, `activation_ramp_epochs` |
| B8 | `sweep_B8_full_curriculum_ac_pairs_bn.json` | 6 | `fraction_acf_local/global`, `activation_ramp_epochs` |
| B9 | `sweep_B9_full_curriculum_global_heavy_bn.json` | 4 | `fraction_dcL/acL`, `activation_ramp_epochs` |
| B10 | `sweep_B10_full_curriculum_local_max_bn.json` | 4 | `fraction_dcG/acG`, `activation_ramp_epochs` |
| B10a | `sweep_B10a_full_curriculum_localDC_globalac_b.json` | 4 | `fraction_acf_global`, `activation_ramp_epochs` |
| C1a | `sweep_C1a_best_runs_residual_dropout.json` | 16 | architecture + flow fractions |
| C1b | `sweep_C1b_best_runs_residual_dropout.json` | 13 | architecture (dropout, residual, activation, hidden_dim) |
| C1c | `sweep_C1c_best_runs_residual_dropout.json` | 16 | architecture + flow fractions (larger values) |

---

## 3. Varied Hyperparameters

### 3.1 GNN Architecture

| Parameter | Code key | Unique values | Sweeps | Notes |
|-----------|----------|---------------|--------|-------|
| Angle prediction mode | `angle_mode` | `edge_delta`, `both` | B1, B4, C1a | `edge_delta` = edge Δθ head only; `both` = dual node+edge heads |
| Hidden dimension | `hidden_dim` | 64, 92, 128 | C1a–C1c | Applies to all GATv2 layers |
| Activation function | `activation` | `leaky_relu`, `gelu` | C1a–C1c | Applied after each conv block |
| Residual connections | `use_residual` | False, True | C1a–C1c | Skip-connection per conv layer |
| Dropout enabled | `use_dropout` | False, True | C1a–C1c | Applied in conv block |
| Dropout rate | `drop_rate` | 0.1, 0.2 | C1a–C1c | Only active when `use_dropout=True` |

**Code location:** `PowerFlowGNN.__init__` — GNN_Powerflow_V2.7_Training.ipynb, code-cell 6 (abs cell ~11).  
`run_hparam_sweep` — code-cell ~30; arguments `hidden_dims`, `angle_modes`, `conv_types`, `use_residual_list`, `use_dropout_list`, `drop_rate_list`.

---

### 3.2 Physics-Informed Loss Fractions

All fractions are **adaptive targets**: the actual weight is computed each epoch as  
`w = fraction × MSE_current / loss_component_current` (capped at `max_w_flow=100`).

| Parameter | Code key | Unique values | Sweeps | Meaning |
|-----------|----------|---------------|--------|---------|
| AC physics fraction | `fraction_physics` | 0.05, 0.10 | B7, C1a, C1c | Target: physics residual = X × MSE |
| PTDF fraction | `fraction_ptdf` | 0.10, 0.20, 0.30 | B2–B3, B5, C1a | Target: PTDF loss = X × MSE |
| DC flow, local | `fraction_dcf_local` | 0.10, 0.20, 0.30 | B1–B3, B5–B7, C1a–C1c | DC P-flow local supervision |
| DC flow, global | `fraction_dcf_global` | 0.00, 0.10, 0.30 | B3, B6, B10, C1a, C1c | DC global flow (0 = disabled) |
| AC flow, local | `fraction_acf_local` | 0.00, 0.10, 0.20, 0.30 | B1–B2, B5–B9, C1a, C1c | AC P+Q local per-edge supervision |
| AC flow, global | `fraction_acf_global` | 0.00, 0.10, 0.20, 0.30 | B4, B6, B8–B10, B10a, C1a, C1c | AC global flow (0 = disabled) |

**Code location:** `PhysicsConfig` dataclass fields `fraction_dcf_local` etc. — V2.7_Training code-cell 9 (abs ~14).  
Adaptive weight computation: `train_power_flow_gnn`, inner epoch loop, calls `compute_flow_loss_dc_local` etc.

---

### 3.3 Activation Schedule (Epoch Offsets)

All epochs are 0-indexed over a **100-epoch** training run.  
The ramp function is `_activation_scale(epoch, start, stop, ramp_epochs)` — linear ramp-up from `start`, optional ramp-down before `stop`.

| Parameter | Code key | Unique values | Notes |
|-----------|----------|---------------|-------|
| Ramp width | `activation_ramp_epochs` | 5, 10, 15 | Epochs for full ramp-up from start |
| Physics activation start | `physics_activation_start` | 60, 70, 80 | Epoch at which physics loss activates |
| PTDF activation start | `ptdf_activation_start` | 40, 60 | Epoch at which PTDF supervision activates |
| PTDF activation stop | `ptdf_activation_stop` | 65, 75, 76, 85 | Epoch after which PTDF weight = 0 |
| DC-local stop | `dcf_local_activation_stop` | 45, 65 | Fixed start = epoch 20 (constant) |
| DC-global start | `dcf_global_activation_start` | None, 40 | None = never activated |
| DC-global stop | `dcf_global_activation_stop` | None, 75 | None = no stop |
| AC-local start | `acf_local_activation_start` | 50, 60, 70 | |
| AC-local stop | `acf_local_activation_stop` | None, 85 | None = no stop |
| AC-global start | `acf_global_activation_start` | None, 80 | None = never activated |

**Code location:** `_activation_scale` — V2.7_Training code-cell 10 (abs ~15); also copied verbatim in V2.7_Analysis for diagnostic plots.  
`run_hparam_sweep` arguments: `physics_activation_list`, `ptdf_activation_list`, `dcf_local_activation_list`, `dcf_global_activation_list`, `acf_local_activation_list`, `acf_global_activation_list`.

---

## 4. Fixed Hyperparameters

| Parameter | Code key | Fixed value | Justification / source |
|-----------|----------|-------------|------------------------|
| Batch size | `batch_size` | 64 | Training throughput |
| Learning rate | `lr` | 1e-4 | Adam optimiser |
| Number of layers | `num_layers` | 3 | Constant in B/C series |
| Conv type | `conv_type` | `gatv2` | GATv2Conv (edge-dim attention) |
| Normalisation | `norm_type` | None | Ablated; not varied in B/C |
| Output head mode | `head_mode` | `with_encoder` | Bus-type-aware output heads |
| Vmag output mode | `vmag_mode` | `residual` | Predict ΔV from flat 1.0 p.u. baseline |
| Random seed | `seed` | 42 | All sweeps |
| Training scenarios | `num_scenarios` | 1500 | `mixed_1500_w` dataset |
| Bus count range | `bus_count_min/max` | 9–35 | Mixed-topology training set |
| Loss weight mode | `loss_weight_mode` | `adaptive` | Fraction-based dynamic weighting |
| Physics variant | `physics_mode` | `rich` | Full PV/slack AC residual |
| P/Q residual weights | `w_P`, `w_Q` | 0.5 / 0.5 | Equal contribution |
| Max adaptive weight cap | `max_w_flow` | 100.0 | All sub-losses |
| Flow supervision target | `flow_target` | `pq` | Both P and Q per-edge |
| DC-local activation start | `dcf_local_activation_start` | 20 | First activation in all runs |
| PTDF loss mode | `ptdf_loss_mode` | `stepwise` | Matrix→flows switch |
| PTDF changepoint | `ptdf_changepoint` | 10 epochs | Relative to PTDF start |
| PTDF mix ratio | `ptdf_alpha` | 0.5 | For stepwise |
| PTDF branch scope | `ptdf_branch_mode` | `all` | Lines + transformers |
| Y-matrix source | `y_matrix_source` | `manual` | Flat-pu convention |
| P-nom share feature | `use_pnom_share` | False | Not used in B/C series |
| Global pooling | `use_global_pool` | False | Ablated |
| PTDF supervision on | `use_ptdf_loss_flag` | True | All B/C runs use PTDF |

---

## 5. Assumptions and Limitations

1. **PTDF always on.** All 98 runs have `use_ptdf_loss_flag=True`. The table does not report runs without PTDF.  
2. **`dcf_local_activation_start` = 20 is fixed.** The DC-local loss is always the first flow loss to activate; only its stop epoch varies.  
3. **`acf_global_activation_start=None`** in runs where `fraction_acf_global=0.0` — these are structurally equivalent (no AC global loss), but the `None` vs integer distinction arises from the sweep config.  
4. **`ptdf_activation_stop` values 75 vs 76** are effectively identical (off-by-one from sweep parameter passing); treated as the same config.  
5. **Normalisation and residual connections** were not varied in B1–B10a; they were introduced only in C1a–C1c.  
6. **`drop_rate`** is only meaningful when `use_dropout=True`; when `use_dropout=False`, `drop_rate` is irrelevant regardless of its value.  
7. **Training time** varies from ~7 h to ~28 h per run (CPU) depending on config; not a controlled variable.  
8. This traceability record covers scalar hyperparameters only. Non-scalar stored artefacts (model weights, `history` dicts, `test_metrics`, `test_networks`) are excluded from the table.
