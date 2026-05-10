---
type: REFACTOR
status: todo
priority: High
effort: 2h
labels: [training, analysis, node-features]
depends-on: []
created: 2026-05-10
completed:
summary: ""
---

# Remove col7 (p_bal_col) from node features + sync Analysis

## Problem

The current 9-feature node layout when `use_pnom_share=True` contains a redundant column:
- **col7** (`p_bal_col`): the total PQ load (`-sum(P_pq)`) broadcast identically to all gen buses. The GNN can compute this internally from PQ node P inputs (col3 + col2 mask), so it wastes a feature dimension.
- **col8** (`p_nom_share`): static `p_nom_i / Σp_nom` per gen bus — genuine input, must keep.

Additionally, the Analysis notebook has diverged from Training: wrong formula (`p_bus.sum()` post-solve ≈ 0), wrong column count (1 combined col instead of 2 separate), and auto-detect threshold `>= 9` that can never trigger on its own 8-feature data.

## Solution

Remove `p_bal_col` from both notebooks. Keep only `p_nom_share_col` (becomes col7). Fix Analysis to match Training. Update auto-detect threshold from `>= 9` to `>= 8`. Add `use_pnom_share` auto-detect to Training's `predict_network_results_with_masks`.

**Why remove rather than fix**: col7 is internally computable — it's a signal amplification hack that adds no information the GNN can't derive from existing features.

## Scope

**Included:**
- Remove `p_bal_col` construction + concatenation in Training `_create_graph_data`
- Rewrite Analysis `_create_graph_data` to match Training target layout
- Change auto-detect threshold `>= 9` → `>= 8` in 3 locations
- Add `use_pnom_share=None` + auto-detect to Training `predict_network_results_with_masks`
- Update docstrings in both notebooks
- Update Training diagnostic cells that reference `x[:, 7]` / `x[:, 8]`
- Remove dead `use_pnet_balance` flag from `PowerFlowDataset.__init__`, `train_power_flow_gnn`, `evaluate_gnn_on_test_set`, and `check_hparam_results` call site (both notebooks, excluding backfill functions)
- Remove `data.p_net_balance` storage + `p_net_balance` local variable computation from `_create_graph_data` in both notebooks
- Remove diagnostic `print(f"p_net_balance: ...")` in Training

**Not included:**
- Analysis backfill functions (`backfill_timing_live`, `backfill_line_flows_live`, `backfill_all_metrics_live`, `evaluate_dc_baseline`) — old runs only, leave untouched including their `use_pnet_balance` params
- Model architecture changes (none needed)

## Affected Files

| File | Change |
|------|--------|
| `GNN_Powerflow_V2.6_Training.ipynb` | `_create_graph_data`: remove p_bal_col + p_net_balance computation + data.p_net_balance storage |
| `GNN_Powerflow_V2.6_Training.ipynb` | `evaluate_gnn_on_test_set`: threshold `>= 9` → `>= 8`, remove `use_pnet_balance` param + passthrough |
| `GNN_Powerflow_V2.6_Training.ipynb` | `predict_network_results_with_masks`: add `use_pnom_share=None` + auto-detect |
| `GNN_Powerflow_V2.6_Training.ipynb` | `PowerFlowDataset.__init__`: remove `use_pnet_balance` param + storage |
| `GNN_Powerflow_V2.6_Training.ipynb` | `train_power_flow_gnn`: remove `use_pnet_balance` param + 3 dataset call passthroughs |
| `GNN_Powerflow_V2.6_Training.ipynb` | Docstring + diagnostic cells: update feature count, col7/col8 refs, remove p_net_balance print |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | `_create_graph_data`: rewrite use_pnom_share block + remove p_net_balance |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | `predict_network_results_with_masks`: threshold `>= 9` → `>= 8` |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | `evaluate_gnn_on_test_set`: threshold `>= 9` → `>= 8`, remove `use_pnet_balance` param + passthrough |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | `PowerFlowDataset.__init__`: remove `use_pnet_balance` param + storage |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | `check_hparam_results`: remove `use_pnet_balance=True` from call site |
| `GNN_Powerflow_V2.6_Analysis.ipynb` | Docstrings: update feature count description |

## Implementation Steps

All changes use the **surgical disk patching** protocol: git checkpoint first, Python patch script, verify with `git diff`.

### 1. Git checkpoint

```bash
git add -A && git commit -m "wip: checkpoint before col7 removal"
```

### 2. Training `_create_graph_data` — remove p_bal_col + p_net_balance (lines 686–705, 839)

**Remove** the entire p_net_balance computation block (lines 686–690) and the `p_bal_col` from the use_pnom_share block. Also remove `data.p_net_balance = p_net_bal_tensor` at line 839.

**Current** (lines 686–705):
```python
        # ── Step 1: p_net_balance — total PQ load (known inputs only) ─────────
        # Sum of PQ bus active power injections (loads). This is a pre-solve
        # quantity that correlates with P_slack without leaking the answer.
        p_net_balance = -float(x_p[pq_mask].sum().item())
        p_net_bal_tensor = torch.tensor([p_net_balance], dtype=torch.float)

        if self.use_pnom_share:
            # Broadcast total PQ load equally to all generator (slack+PV) buses.
            # No p_nom weighting — the GNN learns each bus's participation from
            # cross-snapshot variation (which bus P actually responds to load changes).
            p_bal_col = torch.zeros(n_buses, 1)
            p_nom_share_col = torch.zeros(n_buses, 1)  # col8: static p_nom share per gen bus
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    p_bal_col[bus_to_i[b], 0] = p_net_balance
                    p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom # calculates the share this node has of the total generation capacity in the system, as a static feature to help the GNN learn participation factors
            x = torch.cat([x, p_bal_col, p_nom_share_col], dim=1)
```

**Target** (8 features when True, no p_net_balance):
```python
        if self.use_pnom_share:
            # col7: static p_nom share per gen bus (p_nom_i / Σp_nom)
            # Tells the model each generator's relative capacity for slack distribution.
            p_nom_share_col = torch.zeros(n_buses, 1)
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
            x = torch.cat([x, p_nom_share_col], dim=1)
```

**Also remove** at line 839:
```python
        data.p_net_balance  = p_net_bal_tensor   # Step 1: slack signal
```

**Also remove** nearby docstring/comment at line 851:
```
    Also propagates p_net_balance (scalar per graph, Step 1 slack signal fix).
```

### 3. Training `evaluate_gnn_on_test_set` — threshold (line 3016)

**Current**: `use_pnom_share=(model.node_embedding.in_features >= 9) if use_pnom_share is None else use_pnom_share,  # >=9: 2-col design (col7=pq_load, col8=pnom_share)`

**Target**: `use_pnom_share=(model.node_embedding.in_features >= 8) if use_pnom_share is None else use_pnom_share,  # >=8: 1-col design (col7=pnom_share)`

### 4. Training `predict_network_results_with_masks` — add use_pnom_share (lines 2814–2832)

**Current signature**:
```python
def predict_network_results_with_masks(
    model, network, use_edge_features=True, debug=False,
    # STSI 26.04.07: learned_edge PTDF support
    ptdf_params=None, max_buses=0, ptdf_offset=0,
):
```

**Target signature**:
```python
def predict_network_results_with_masks(
    model, network, use_edge_features=True, debug=False, use_pnom_share=None,
    # STSI 26.04.07: learned_edge PTDF support
    ptdf_params=None, max_buses=0, ptdf_offset=0,
):
```

**Current dataset creation**:
```python
    dataset = PowerFlowDataset([network], use_edge_features=use_edge_features)
```

**Target dataset creation**:
```python
    dataset = PowerFlowDataset(
        [network],
        use_edge_features=use_edge_features,
        use_pnom_share=(model.node_embedding.in_features >= 8) if use_pnom_share is None else use_pnom_share,
    )
```

### 5. Training docstring (line 567)

**Current**: `x         : [n_buses, 7 or 8] node features (bus type flags, known P/Q/V, optional p_net_balance)`

**Target**: `x         : [n_buses, 7 or 8] node features (bus type flags, known P/Q/V/Vang, optional p_nom_share)`

### 6. Analysis `_create_graph_data` — rewrite use_pnom_share block + remove p_net_balance (lines 709–728, 862)

**Current** (DIVERGED — wrong formula + wrong semantics + p_net_balance stored):
```python
        # ── Step 1: p_net_balance — graph-level scalar for slack signal ───────
        # Net active power balance = sum of all bus injections at this snapshot.
        # This gives the slack bus a varying signal correlated with its true P.
        p_net_balance = float(p_bus.sum().item())
        p_net_bal_tensor = torch.tensor([p_net_balance], dtype=torch.float)

        if self.use_pnom_share:
            # Participation-weighted balance: (p_nom_i / sum_p_nom) * p_net_balance
            # Gives each generator its expected MW contribution to the total imbalance.
            # For PQ nodes: 0.0 (they do not participate in slack compensation).
            # NOTE: use_pnet_balance flag is superseded — use_pnom_share controls this.
            p_nom_col = torch.zeros(n_buses, 1)
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    share = gen["p_nom"] / total_p_nom
                    p_nom_col[bus_to_i[b], 0] += share * p_net_balance
            x = torch.cat([x, p_nom_col], dim=1)
```

**Target** (matches Training — no p_net_balance, just p_nom_share_col):
```python
        if self.use_pnom_share:
            # col7: static p_nom share per gen bus (p_nom_i / Σp_nom)
            # Tells the model each generator's relative capacity for slack distribution.
            p_nom_share_col = torch.zeros(n_buses, 1)
            p_noms = network.generators["p_nom"].values.astype(float)
            total_p_nom = float(p_noms.sum()) if p_noms.sum() > 0 else 1.0
            for gen_name, gen in network.generators.iterrows():
                b = gen["bus"]
                if b in bus_to_i:
                    p_nom_share_col[bus_to_i[b], 0] += gen["p_nom"] / total_p_nom
            x = torch.cat([x, p_nom_share_col], dim=1)
```

**Also remove** at line 862:
```python
        data.p_net_balance  = p_net_bal_tensor   # Step 1: slack signal
```

### 7. Analysis auto-detect thresholds — `>= 9` → `>= 8` (lines 1087, 1699)

Same pattern as Training step 3. Two locations:
- `predict_network_results_with_masks` line 1087
- `evaluate_gnn_on_test_set` line 1699

Both change from:
```python
use_pnom_share=(model.node_embedding.in_features >= 9) if use_pnom_share is None else use_pnom_share,  # >=9: 2-col design (col7=pq_load, col8=pnom_share)
```
to:
```python
use_pnom_share=(model.node_embedding.in_features >= 8) if use_pnom_share is None else use_pnom_share,  # >=8: 1-col design (col7=pnom_share)
```

### 8. Analysis docstrings (lines 590, 604)

**Line 590 current**: `x         : [n_buses, 7 or 8] node features (bus type flags, known P/Q/V, optional p_net_balance)`

**Target**: `x         : [n_buses, 7 or 8] node features (bus type flags, known P/Q/V/Vang, optional p_nom_share)`

**Line 604 current**: `use_pnet_balance: bool = True,   # Step 1: append p_net_balance to node features`

Keep as-is (dead flag removal is deferred).

### 9. Remove dead `use_pnet_balance` flag — Training (lines 581, 590, 2024, 2073, 2080, 2087, 2972, 3015)

**`PowerFlowDataset.__init__`** — remove param and storage:
- Line 581: delete `use_pnet_balance: bool = True,   # Step 1: append p_net_balance to node features`
- Line 590: delete `self.use_pnet_balance  = use_pnet_balance`

**`train_power_flow_gnn`** — remove param and 3 passthroughs:
- Line 2024: delete `use_pnet_balance=True, ...` parameter line
- Lines 2073, 2080, 2087: delete `use_pnet_balance=use_pnet_balance, ...` from 3 `PowerFlowDataset(` calls

**`evaluate_gnn_on_test_set`** — remove param and passthrough:
- Line 2972: delete `use_pnet_balance: bool = True,` parameter line
- Line 3015: delete `use_pnet_balance=use_pnet_balance,` from `PowerFlowDataset(` call

**Diagnostic print** — remove:
- Line 2096: delete `print(f"p_net_balance: {sample.p_net_balance}")   # must be non-zero`

### 10. Remove dead `use_pnet_balance` flag — Analysis (lines 604, 613, 1652, 1698, 2953)

**`PowerFlowDataset.__init__`** — remove param and storage:
- Line 604: delete `use_pnet_balance: bool = True,   # Step 1: append p_net_balance to node features`
- Line 613: delete `self.use_pnet_balance  = use_pnet_balance`

**`evaluate_gnn_on_test_set`** — remove param and passthrough:
- Line 1652: delete `use_pnet_balance: bool = True,` parameter line
- Line 1698: delete `use_pnet_balance=use_pnet_balance,` from `PowerFlowDataset(` call

**`check_hparam_results` call site** — remove kwarg:
- Line 2953: delete `use_pnet_balance=True,` from `evaluate_gnn_on_test_set(` call

**DO NOT TOUCH** backfill functions (`backfill_timing_live`, `backfill_line_flows_live`, `backfill_all_metrics_live`, `evaluate_dc_baseline`) — they will still pass `use_pnet_balance` to `PowerFlowDataset`, but since the param was removed from `__init__`, Python will raise a `TypeError`. To avoid this, change `PowerFlowDataset.__init__` to accept `**kwargs` as a catch-all instead of deleting the param — OR leave `use_pnet_balance` in `__init__` as ignored.

**Decision**: Keep `use_pnet_balance` in `PowerFlowDataset.__init__` signature in both notebooks (as a no-op param that is accepted but ignored). This avoids breaking backfill functions while removing the dead flag from all other functions.

### 11. Training diagnostic cells (lines 8832–8975)

These are exploratory cells with hardcoded `x[:, 7]` and `x[:, 8]`. After the change:
- `x[:, 7]` = p_nom_share (was p_bal_col)
- `x[:, 8]` = doesn't exist (was p_nom_share_col)

Update references:
- `data.x[:, 7]` stays but now means p_nom_share
- `data.x[:, 8]` references → remove or change to `data.x[:, 7]`
- Comments like `col7=...  col8=...` → `col7=pnom_share` only
- `n_features=9` → `n_features=8`

### 12. Verify with git diff

After all patches, run `git diff` and confirm:
- No `p_bal_col` references remain in core code
- All `>= 9` auto-detect changed to `>= 8`
- Analysis `_create_graph_data` no longer uses `p_bus.sum()` or `p_net_balance`
- Training `predict_network_results_with_masks` has `use_pnom_share` parameter
- `use_pnet_balance` removed from all non-backfill function signatures
- `data.p_net_balance` storage removed from both notebooks
- `PowerFlowDataset.__init__` still accepts `use_pnet_balance` (ignored) to avoid breaking backfill

## Acceptance Criteria

- [ ] Training `_create_graph_data` produces 8 features (not 9) when `use_pnom_share=True`
- [ ] Analysis `_create_graph_data` produces 8 features and matches Training logic exactly (no p_net_balance)
- [ ] Auto-detect threshold is `>= 8` in all 3 locations
- [ ] Training `predict_network_results_with_masks` accepts `use_pnom_share=None` with auto-detect
- [ ] `data.p_net_balance` is no longer stored in either notebook
- [ ] `use_pnet_balance` removed from `train_power_flow_gnn`, `evaluate_gnn_on_test_set` (both notebooks), and `check_hparam_results` call site
- [ ] `PowerFlowDataset.__init__` still accepts `use_pnet_balance` (ignored) so backfill functions don't break
- [ ] Old 7-feature models still work (base features unaffected)
- [ ] `git diff` shows no unintended changes
- [ ] Both notebooks parse as valid JSON after patching
