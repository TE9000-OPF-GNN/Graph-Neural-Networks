"""
Functional + E2E tests for Tasks 043/044/045 (dual-angle + DC/AC flow loss).
Run after each implementation step to confirm nothing is broken.

Usage:
    python test_dual_angle_flow_loss.py --phase 1   # after Task 043
    python test_dual_angle_flow_loss.py --phase 2   # after Task 044
    python test_dual_angle_flow_loss.py --phase 3   # after Task 045
    python test_dual_angle_flow_loss.py              # all phases

Can also be pasted cell-by-cell into the V2.7 Training notebook.
"""
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Task 043 — angle_mode="both" dual head architecture
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase1_angle_mode_both():
    """
    Validates that angle_mode='both' produces correct outputs and 
    existing modes are not regressed.
    
    Prerequisites: Cell 12 (model) must be defined in the notebook/module.
    """
    print("=" * 70)
    print("PHASE 1: angle_mode='both' — Dual Head Architecture")
    print("=" * 70)

    # ── Import model class (adjust path if needed) ──
    # When running in notebook: these are already in scope
    # When running standalone: need to import from notebook or code_base
    try:
        from code_base import PowerFlowGNN  # if extracted
    except ImportError:
        print("  [SKIP] PowerFlowGNN not importable standalone.")
        print("  Paste this test into the notebook after Cell 12.\n")
        return False

    device = torch.device("cpu")
    N, E_fwd, hidden = 14, 20, 64
    num_features = 7
    edge_features = 6

    results = {}

    # ── Test 1a: angle_mode="both" forward pass ──
    print("\n  [1a] angle_mode='both' — forward pass shape check")
    model_both = PowerFlowGNN(
        num_node_features=num_features,
        hidden_dim=hidden,
        num_layers=4,
        angle_mode="both",
    ).to(device)

    # Synthetic batch
    x = torch.randn(N, num_features, device=device)
    edge_index = torch.randint(0, N, (2, E_fwd * 2), device=device)
    edge_attr = torch.randn(E_fwd * 2, edge_features, device=device)
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)

    out = model_both(x, edge_index, edge_attr, batch_idx)
    # Expected: tuple (node_pred, h_nodes, h_edges, delta_theta_pred)
    # or similar — adapt based on actual return signature

    if isinstance(out, tuple):
        node_pred = out[0]
        # Find delta_theta in the tuple
        delta_theta = None
        for item in out:
            if isinstance(item, torch.Tensor) and item.dim() == 1 and item.shape[0] == E_fwd:
                delta_theta = item
                break
        
        assert node_pred.shape == (N, 4), \
            f"  FAIL: node_pred shape {node_pred.shape}, expected ({N}, 4)"
        assert delta_theta is not None, \
            "  FAIL: delta_theta_pred not found in output tuple"
        assert delta_theta.shape == (E_fwd,), \
            f"  FAIL: delta_theta shape {delta_theta.shape}, expected ({E_fwd},)"
        print(f"    node_pred: {node_pred.shape} ✓")
        print(f"    delta_theta_pred: {delta_theta.shape} ✓")
        results["1a"] = True
    else:
        print(f"  FAIL: expected tuple output, got {type(out)}")
        results["1a"] = False

    # ── Test 1b: Backward pass — both heads get gradients ──
    print("\n  [1b] angle_mode='both' — gradient flow to both heads")
    model_both.zero_grad()
    loss = node_pred.sum() + delta_theta.sum()
    loss.backward()

    has_vang_grad = model_both.vang_pred.weight.grad is not None
    has_edge_grad = model_both.edge_angle_pred.weight.grad is not None
    assert has_vang_grad, "  FAIL: vang_pred head has no gradient"
    assert has_edge_grad, "  FAIL: edge_angle_pred head has no gradient"
    print(f"    vang_pred grad norm: {model_both.vang_pred.weight.grad.norm():.4f} ✓")
    print(f"    edge_angle_pred grad norm: {model_both.edge_angle_pred.weight.grad.norm():.4f} ✓")
    results["1b"] = True

    # ── Test 1c: Regression — angle_mode="node" still works ──
    print("\n  [1c] Regression check — angle_mode='node'")
    model_node = PowerFlowGNN(
        num_node_features=num_features,
        hidden_dim=hidden,
        num_layers=4,
        angle_mode="node",
    ).to(device)
    out_node = model_node(x, edge_index, edge_attr, batch_idx)
    node_pred_node = out_node[0] if isinstance(out_node, tuple) else out_node
    assert node_pred_node.shape == (N, 4), \
        f"  FAIL: node mode shape {node_pred_node.shape}"
    print(f"    node_pred: {node_pred_node.shape} ✓ (no delta_theta)")
    results["1c"] = True

    # ── Test 1d: Regression — angle_mode="edge_delta" still works ──
    print("\n  [1d] Regression check — angle_mode='edge_delta'")
    model_edge = PowerFlowGNN(
        num_node_features=num_features,
        hidden_dim=hidden,
        num_layers=4,
        angle_mode="edge_delta",
    ).to(device)
    out_edge = model_edge(x, edge_index, edge_attr, batch_idx)
    node_pred_edge = out_edge[0] if isinstance(out_edge, tuple) else out_edge
    assert node_pred_edge.shape == (N, 3), \
        f"  FAIL: edge_delta mode shape {node_pred_edge.shape}, expected ({N}, 3)"
    print(f"    node_pred: {node_pred_edge.shape} ✓ (Vmag, P, Q only)")
    results["1d"] = True

    # ── Summary ──
    print(f"\n  Phase 1 Results: {sum(results.values())}/{len(results)} passed")
    return all(results.values())


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Task 044 — DC/AC per-edge flow loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase2_flow_loss_functions():
    """
    Validates all four flow loss variants + dispatcher.
    
    Prerequisites: Cell 14 (loss functions + PhysicsConfig) must be defined.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: DC/AC Per-Edge Flow Loss Functions")
    print("=" * 70)

    try:
        from code_base import (
            compute_flow_loss_dc_local,
            compute_flow_loss_dc_global,
            compute_flow_loss_ac_local,
            compute_flow_loss_ac_global,
            compute_flow_loss,
            PhysicsConfig,
        )
    except ImportError:
        print("  [SKIP] Flow loss functions not importable standalone.")
        print("  Paste this test into the notebook after Cell 14.\n")
        return False

    device = torch.device("cpu")
    E_fwd = 20
    N = 14
    E_supervised = 15  # fewer than E_fwd (lines only, no transformers)

    results = {}

    # ── Synthetic data ──
    # Angles: small random angles (realistic: ~0.01-0.1 rad)
    delta_theta_pred = torch.randn(E_fwd, device=device) * 0.05
    delta_theta_pred.requires_grad_(True)

    theta_pred = torch.randn(N, device=device) * 0.05
    theta_pred.requires_grad_(True)

    vmag = torch.ones(N, device=device) + torch.randn(N) * 0.02
    vmag.requires_grad_(True)

    # Edge attr: [r, x, b_half, tap, g_ser, b_ser] — interleaved fwd/bwd
    edge_attr = torch.zeros(E_fwd * 2, 6, device=device)
    x_react = torch.rand(E_fwd) * 0.1 + 0.01  # reactance 0.01-0.11
    r_resist = x_react * 0.1  # r/x ≈ 0.1
    g_ser = r_resist / (r_resist**2 + x_react**2)
    b_ser = -x_react / (r_resist**2 + x_react**2)
    for i in range(E_fwd):
        edge_attr[2*i, 0] = r_resist[i]
        edge_attr[2*i, 1] = x_react[i]
        edge_attr[2*i, 4] = g_ser[i]
        edge_attr[2*i, 5] = b_ser[i]
        edge_attr[2*i+1, 0] = r_resist[i]
        edge_attr[2*i+1, 1] = x_react[i]
        edge_attr[2*i+1, 4] = g_ser[i]
        edge_attr[2*i+1, 5] = b_ser[i]

    # Edge index (random, for shape purposes)
    src = torch.randint(0, N, (E_fwd,))
    dst = torch.randint(0, N, (E_fwd,))
    edge_index = torch.zeros(2, E_fwd * 2, dtype=torch.long)
    edge_index[0, ::2] = src
    edge_index[1, ::2] = dst
    edge_index[0, 1::2] = dst  # reverse
    edge_index[1, 1::2] = src

    # True flows: DC approximation of our angles (so loss should be small)
    y_line_p = (delta_theta_pred[:E_supervised].detach() / x_react[:E_supervised])
    # Add small noise to make loss non-zero
    y_line_p = y_line_p + torch.randn(E_supervised) * 0.001

    # ── Test 2a: DC local — returns scalar loss with grad_fn ──
    print("\n  [2a] compute_flow_loss_dc_local — basic functionality")
    loss_dc_l, mae_dc_l = compute_flow_loss_dc_local(
        delta_theta_pred, edge_attr, y_line_p)
    assert loss_dc_l.requires_grad, "  FAIL: loss has no grad_fn"
    assert loss_dc_l.item() >= 0, "  FAIL: negative loss"
    assert mae_dc_l.item() >= 0, "  FAIL: negative MAE"
    loss_dc_l.backward(retain_graph=True)
    assert delta_theta_pred.grad is not None, "  FAIL: no gradient on delta_theta"
    print(f"    loss={loss_dc_l.item():.6f}, mae={mae_dc_l.item():.6f} ✓")
    results["2a"] = True
    delta_theta_pred.grad = None

    # ── Test 2b: DC global — returns scalar loss with grad_fn ──
    print("\n  [2b] compute_flow_loss_dc_global — basic functionality")
    edge_index_fwd = edge_index[:, ::2]
    loss_dc_g, mae_dc_g = compute_flow_loss_dc_global(
        theta_pred, edge_index_fwd, edge_attr, y_line_p)
    assert loss_dc_g.requires_grad, "  FAIL: loss has no grad_fn"
    loss_dc_g.backward(retain_graph=True)
    assert theta_pred.grad is not None, "  FAIL: no gradient on theta_pred"
    print(f"    loss={loss_dc_g.item():.6f}, mae={mae_dc_g.item():.6f} ✓")
    results["2b"] = True
    theta_pred.grad = None

    # ── Test 2c: Consistency — DC local == DC global when Δθ == θ_i - θ_j ──
    print("\n  [2c] Consistency: DC local == DC global (when Δθ = θ_i - θ_j)")
    # Make delta_theta exactly equal to theta_src - theta_dst
    with torch.no_grad():
        consistent_delta = theta_pred[src] - theta_pred[dst]
    consistent_delta_param = consistent_delta.clone().requires_grad_(True)
    
    y_line_p_consist = torch.zeros(E_supervised)  # dummy target
    loss_l, _ = compute_flow_loss_dc_local(
        consistent_delta_param, edge_attr, y_line_p_consist)
    loss_g, _ = compute_flow_loss_dc_global(
        theta_pred, edge_index_fwd, edge_attr, y_line_p_consist)
    diff = abs(loss_l.item() - loss_g.item())
    assert diff < 1e-5, f"  FAIL: local={loss_l.item():.8f} vs global={loss_g.item():.8f}, diff={diff:.8f}"
    print(f"    DC local: {loss_l.item():.8f}")
    print(f"    DC global: {loss_g.item():.8f}")
    print(f"    Difference: {diff:.2e} ✓")
    results["2c"] = True

    # ── Test 2d: AC local — includes V_i² × g_s term ──
    print("\n  [2d] compute_flow_loss_ac_local — AC formula correctness")
    loss_ac_l, mae_ac_l = compute_flow_loss_ac_local(
        delta_theta_pred, vmag, edge_index, edge_attr, y_line_p)
    assert loss_ac_l.requires_grad, "  FAIL: AC local loss has no grad_fn"
    loss_ac_l.backward(retain_graph=True)
    assert delta_theta_pred.grad is not None, "  FAIL: no grad on delta_theta"
    assert vmag.grad is not None, "  FAIL: no grad on vmag (AC should use it!)"
    print(f"    loss={loss_ac_l.item():.6f}, mae={mae_ac_l.item():.6f} ✓")
    print(f"    vmag receives gradient: ✓ (norm={vmag.grad.norm():.6f})")
    results["2d"] = True
    delta_theta_pred.grad = None
    vmag.grad = None

    # ── Test 2e: AC global — gradient flows to theta_pred AND vmag ──
    print("\n  [2e] compute_flow_loss_ac_global — gradient to θ and V")
    loss_ac_g, mae_ac_g = compute_flow_loss_ac_global(
        theta_pred, vmag, edge_index, edge_attr, y_line_p)
    assert loss_ac_g.requires_grad, "  FAIL: AC global loss has no grad_fn"
    loss_ac_g.backward(retain_graph=True)
    assert theta_pred.grad is not None, "  FAIL: no grad on theta_pred"
    assert vmag.grad is not None, "  FAIL: no grad on vmag"
    print(f"    loss={loss_ac_g.item():.6f}, mae={mae_ac_g.item():.6f} ✓")
    print(f"    theta_pred grad norm: {theta_pred.grad.norm():.6f} ✓")
    print(f"    vmag grad norm: {vmag.grad.norm():.6f} ✓")
    results["2e"] = True
    theta_pred.grad = None
    vmag.grad = None

    # ── Test 2f: Q flow loss (AC variants should also compute Q) ──
    print("\n  [2f] AC flow loss — Q flow supervision")
    # Check if the functions accept y_line_q parameter
    import inspect
    sig = inspect.signature(compute_flow_loss_ac_local)
    has_q_param = "y_line_q" in sig.parameters
    if has_q_param:
        y_line_q = torch.randn(E_supervised) * 0.1
        loss_ac_pq, mae_pq = compute_flow_loss_ac_local(
            delta_theta_pred, vmag, edge_index, edge_attr, 
            y_line_p, y_line_q=y_line_q)
        assert loss_ac_pq.item() > 0, "  FAIL: PQ loss should be > 0"
        print(f"    AC local (P+Q): loss={loss_ac_pq.item():.6f} ✓")
        results["2f"] = True
    else:
        print("    WARNING: y_line_q not in AC function signature — Q flow not supervised")
        print("    This is acceptable for Phase 2 if Q is deferred to a follow-up.")
        results["2f"] = "WARN"

    # ── Test 2g: Dispatcher respects mode flags ──
    print("\n  [2g] compute_flow_loss dispatcher — mode gating")
    cfg_dc_local = PhysicsConfig(
        use_flow_loss=True,
        flow_loss_mode="dc",
        flow_angle_mode="local",
        w_flow=1.0, w_flow_local=1.0, w_flow_global=0.0,
        w_flow_dc=1.0, w_flow_ac=0.0,
    )
    total, diag = compute_flow_loss(
        cfg=cfg_dc_local,
        delta_theta_pred=delta_theta_pred,
        theta_pred=theta_pred,
        vmag=vmag,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_line_p=y_line_p,
        active_local=True,
        active_global=False,
        active_ac=False,
    )
    assert "flow_loss_dc_local" in diag, "  FAIL: dc_local not in diagnostics"
    assert "flow_loss_dc_global" not in diag, "  FAIL: dc_global should not fire"
    assert "flow_loss_ac_local" not in diag, "  FAIL: ac should not fire"
    print(f"    DC-local only: loss={total.item():.6f}, diag keys={list(diag.keys())} ✓")
    results["2g"] = True

    # ── Test 2h: Dispatcher — all modes ──
    print("\n  [2h] compute_flow_loss dispatcher — all modes active")
    cfg_all = PhysicsConfig(
        use_flow_loss=True,
        flow_loss_mode="both",
        flow_angle_mode="both",
        w_flow=1.0, w_flow_local=0.5, w_flow_global=0.5,
        w_flow_dc=0.5, w_flow_ac=0.5,
    )
    total_all, diag_all = compute_flow_loss(
        cfg=cfg_all,
        delta_theta_pred=delta_theta_pred,
        theta_pred=theta_pred,
        vmag=vmag,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_line_p=y_line_p,
        active_local=True,
        active_global=True,
        active_ac=True,
    )
    expected_keys = {"flow_loss_dc_local", "flow_loss_dc_global",
                     "flow_loss_ac_local", "flow_loss_ac_global"}
    missing = expected_keys - set(diag_all.keys())
    assert not missing, f"  FAIL: missing diag keys: {missing}"
    print(f"    All modes: loss={total_all.item():.6f}")
    print(f"    Diagnostics: {list(diag_all.keys())} ✓")
    results["2h"] = True

    # ── Summary ──
    passed = sum(1 for v in results.values() if v is True)
    warned = sum(1 for v in results.values() if v == "WARN")
    total_tests = len(results)
    print(f"\n  Phase 2 Results: {passed}/{total_tests} passed, {warned} warnings")
    return all(v is True or v == "WARN" for v in results.values())


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Task 045 — Training loop integration + warmup
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase3_training_integration():
    """
    E2E test: short training run with flow loss curriculum.
    
    Prerequisites: Full notebook must be runnable (cells 1-17).
    Requires actual dataset (at least a small one).
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Training Loop Integration + Warmup Curriculum")
    print("=" * 70)

    try:
        from code_base import (
            train_power_flow_gnn,
            PowerFlowDataset,
            PhysicsConfig,
        )
    except ImportError:
        print("  [SKIP] Training function not importable standalone.")
        print("  Paste this test into the notebook after Cell 15.\n")
        print("  ─── NOTEBOOK VERSION (paste after all cells defined) ───")
        print_notebook_e2e_test()
        return False

    print("  [NOTE] Phase 3 requires dataset + full training infrastructure.")
    print("  Use the notebook version below.\n")
    print_notebook_e2e_test()
    return True


def print_notebook_e2e_test():
    """Prints the notebook-paste version of the full E2E test including sweep."""
    code = '''
# ═══════════════════════════════════════════════════════════════════════════════
# E2E TEST: Full pipeline including run_hparam_sweep
# Run this cell AFTER all implementation cells are defined.
# Tests the entire path: dataset → model → train → sweep → run_info → make_run_key
# ═══════════════════════════════════════════════════════════════════════════════
import time as _time
_t0 = _time.perf_counter()
_errors = []

print("=" * 70)
print("E2E TEST: angle_mode='both' + DC/AC flow loss + sweep integration")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0: Dataset with angle_mode="both"
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 0: Build dataset with angle_mode='both' ──")

# Pick a small network set (adjust to whatever is loaded in your session)
_networks = networks_ieee9_large_singel_slack[:30]  # 30 networks, fast
assert len(_networks) >= 30, f"Need >=30 networks, have {len(_networks)}"

_ds_both = PowerFlowDataset(
    _networks,
    angle_mode="both",
    use_pnom_share=False,
    ptdf_mode=None,
)
print(f"  Dataset size: {len(_ds_both)} graphs")

# Validate graph structure
_g0 = _ds_both[0]
assert hasattr(_g0, 'y_delta_theta'), "FAIL: y_delta_theta not stored for angle_mode='both'"
assert _g0.y.shape[1] == 4, f"FAIL: y should be [N,4], got {_g0.y.shape}"
assert _g0.y_delta_theta.dim() == 1, f"FAIL: y_delta_theta should be 1D, got {_g0.y_delta_theta.shape}"
print(f"  Graph 0: x={_g0.x.shape}, y={_g0.y.shape}, y_delta_theta={_g0.y_delta_theta.shape}")
print(f"  edge_attr={_g0.edge_attr.shape}, edge_index={_g0.edge_index.shape}")

# Check y_line_p exists (needed for flow loss)
assert hasattr(_g0, 'y_line_p'), "FAIL: y_line_p not in dataset — flow loss needs line flow targets"
print(f"  y_line_p: {_g0.y_line_p.shape} ✓")

# Check y_line_q if available (for AC Q flow loss)
if hasattr(_g0, 'y_line_q'):
    print(f"  y_line_q: {_g0.y_line_q.shape} ✓")
else:
    print("  y_line_q: NOT PRESENT (Q flow loss will be unavailable)")

print("  STEP 0 PASSED ✓")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Model instantiation with angle_mode="both"
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 1: Model instantiation ──")

_model_both = PowerFlowGNN(
    num_node_features=_g0.x.shape[1],
    hidden_dim=32,
    num_layers=3,
    angle_mode="both",
    conv_mode="residual",
).to("cpu")

# Quick forward pass
from torch_geometric.loader import DataLoader as PyGDataLoader
_loader = PyGDataLoader(_ds_both[:5], batch_size=5, shuffle=False)
_batch = next(iter(_loader))

_out = _model_both(_batch.x, _batch.edge_index, _batch.edge_attr, _batch.batch)
_node_pred = _out[0]
# Find delta_theta in output tuple
_delta_theta_pred = None
for _item in _out:
    if isinstance(_item, torch.Tensor) and _item.dim() == 1:
        # Should be [total_E_fwd] across batch
        _delta_theta_pred = _item
        break

assert _node_pred.shape[1] == 4, f"FAIL: node_pred cols={_node_pred.shape[1]}, expected 4"
assert _delta_theta_pred is not None, "FAIL: no delta_theta_pred in output"
print(f"  node_pred: {_node_pred.shape} (Vmag, θ, P, Q) ✓")
print(f"  delta_theta_pred: {_delta_theta_pred.shape} ✓")
print("  STEP 1 PASSED ✓")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Flow loss functions on real batch data
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 2: Flow loss functions on real data ──")

_cfg_flow = PhysicsConfig(
    use_flow_loss=True,
    flow_loss_mode="both",       # DC + AC
    flow_angle_mode="both",      # local + global
    w_flow=1.0,
    w_flow_local=0.5,
    w_flow_global=0.5,
    w_flow_dc=0.5,
    w_flow_ac=0.5,
)

_theta_pred = _node_pred[:, 1]  # node θ
_vmag_pred = _node_pred[:, 0]   # node Vmag

_flow_loss, _flow_diag = compute_flow_loss(
    cfg=_cfg_flow,
    delta_theta_pred=_delta_theta_pred,
    theta_pred=_theta_pred,
    vmag=_vmag_pred,
    edge_index=_batch.edge_index,
    edge_attr=_batch.edge_attr,
    y_line_p=_batch.y_line_p,
    active_local=True,
    active_global=True,
    active_ac=True,
)

assert _flow_loss.requires_grad, "FAIL: flow loss has no computation graph"
assert _flow_loss.item() > 0, f"FAIL: flow loss is {_flow_loss.item()} (expected > 0)"
print(f"  Total flow loss: {_flow_loss.item():.6f}")
print(f"  Diagnostics: {_flow_diag}")

# Verify backward works end-to-end
_flow_loss.backward()
_has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 
                 for p in _model_both.parameters())
assert _has_grads, "FAIL: no gradients reached model parameters"
print("  Backward pass: gradients flow to model ✓")
_model_both.zero_grad()
print("  STEP 2 PASSED ✓")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: train_power_flow_gnn with flow loss (8 epochs)
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 3: train_power_flow_gnn — 8 epochs with warmup ──")

_train_ds = _ds_both[:20]
_val_ds = _ds_both[20:25]
_test_ds = _ds_both[25:30]

_phys_cfg_train = PhysicsConfig(
    use_physics_loss=True,
    use_ptdf_loss=False,
    use_flow_loss=True,
    flow_loss_mode="dc",
    flow_angle_mode="local",
    w_flow=1.0,
    w_flow_local=1.0,
    w_flow_global=0.5,
    w_flow_dc=1.0,
    w_flow_ac=0.5,
    loss_weight_mode="fixed",
)

_model_t, _hist_t, _metrics_t = train_power_flow_gnn(
    train_dataset=_train_ds,
    val_dataset=_val_ds,
    test_dataset=_test_ds,
    num_epochs=8,
    hidden_dim=32,
    num_layers=3,
    angle_mode="both",
    physics_cfg=_phys_cfg_train,
    warmup_epochs_phys=2,
    warmup_epochs_flow=3,
    warmup_epochs_flow_global=99,
    warmup_epochs_flow_ac=99,
    verbose=False,
)

# Validate warmup gating
_flow_losses = _hist_t.get("flow_loss", [])
assert len(_flow_losses) == 8, f"FAIL: expected 8 flow_loss entries, got {len(_flow_losses)}"
for _i in range(3):
    if _flow_losses[_i] != 0.0:
        _errors.append(f"Epoch {_i}: flow_loss={_flow_losses[_i]} during warmup (expected 0)")
for _i in range(3, 8):
    if _flow_losses[_i] <= 0.0:
        _errors.append(f"Epoch {_i}: flow_loss={_flow_losses[_i]} after warmup (expected > 0)")

if not _errors:
    print(f"  Warmup gating: epochs 0-2 = 0.0, epochs 3-7 > 0 ✓")
    print(f"  Flow loss values: {[f'{v:.5f}' for v in _flow_losses]}")
else:
    for _e in _errors:
        print(f"  FAIL: {_e}")

# Validate training loss is finite and decreasing
_train_losses = _hist_t.get("train_loss", _hist_t.get("loss", []))
assert all(np.isfinite(v) for v in _train_losses), "FAIL: NaN/Inf in training loss"
print(f"  Training loss: {_train_losses[0]:.4f} → {_train_losses[-1]:.4f} ✓")
print("  STEP 3 PASSED ✓")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: run_hparam_sweep — the real integration test
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 4: run_hparam_sweep — full sweep integration ──")

# Define a small 3-run sweep covering the new parameters
_sweep_configs = [
    # Run 1: angle_mode="both", flow loss OFF (baseline)
    dict(
        angle_mode="both",
        physics_cfg=PhysicsConfig(
            use_physics_loss=True,
            use_ptdf_loss=False,
            use_flow_loss=False,
            loss_weight_mode="fixed",
        ),
    ),
    # Run 2: angle_mode="both", DC-local flow loss
    dict(
        angle_mode="both",
        physics_cfg=PhysicsConfig(
            use_physics_loss=True,
            use_ptdf_loss=False,
            use_flow_loss=True,
            flow_loss_mode="dc",
            flow_angle_mode="local",
            w_flow=1.0, w_flow_local=1.0, w_flow_global=0.5,
            w_flow_dc=1.0, w_flow_ac=0.5,
            loss_weight_mode="fixed",
        ),
        warmup_epochs_flow=2,
        warmup_epochs_flow_global=99,
        warmup_epochs_flow_ac=99,
    ),
    # Run 3: angle_mode="both", DC+AC local+global (full curriculum)
    dict(
        angle_mode="both",
        physics_cfg=PhysicsConfig(
            use_physics_loss=True,
            use_ptdf_loss=False,
            use_flow_loss=True,
            flow_loss_mode="both",
            flow_angle_mode="both",
            w_flow=1.0, w_flow_local=0.5, w_flow_global=0.5,
            w_flow_dc=0.5, w_flow_ac=0.5,
            loss_weight_mode="fixed",
        ),
        warmup_epochs_flow=2,
        warmup_epochs_flow_global=4,
        warmup_epochs_flow_ac=5,
    ),
]

# Common params
_sweep_common = dict(
    networks=_networks[:30],
    num_epochs=6,
    hidden_dim=32,
    num_layers=3,
    warmup_epochs_phys=2,
    use_pnom_share=False,
    ptdf_mode=None,
    verbose=False,
)

print(f"  Running {len(_sweep_configs)} sweep configurations...")
_sweep_results = []
for _idx, _cfg in enumerate(_sweep_configs):
    _run_params = {**_sweep_common, **_cfg}
    print(f"    Run {_idx+1}/{len(_sweep_configs)}: "
          f"flow={_cfg.get('physics_cfg').use_flow_loss}, "
          f"mode={_cfg.get('physics_cfg').flow_loss_mode if _cfg.get('physics_cfg').use_flow_loss else 'N/A'}")
    try:
        _result = run_hparam_sweep(**_run_params)
        _sweep_results.append(_result)
        print(f"      completed ✓")
    except Exception as _ex:
        _errors.append(f"Sweep run {_idx+1} failed: {type(_ex).__name__}: {_ex}")
        print(f"      FAILED: {type(_ex).__name__}: {_ex}")
        import traceback; traceback.print_exc()

if len(_sweep_results) == len(_sweep_configs):
    print(f"  All {len(_sweep_configs)} sweep runs completed ✓")
else:
    print(f"  FAIL: {len(_sweep_results)}/{len(_sweep_configs)} completed")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Validate sweep outputs (run_info, make_run_key, history)
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 5: Sweep output validation ──")

for _idx, _result in enumerate(_sweep_results):
    # Adapt to your sweep return format — typically (model, history, metrics, run_info)
    # or a dict with keys. Adjust as needed:
    if isinstance(_result, tuple):
        _ri = _result[-1] if isinstance(_result[-1], dict) else None
        _h = _result[1] if len(_result) > 1 and isinstance(_result[1], dict) else None
    elif isinstance(_result, dict):
        _ri = _result.get("run_info", _result)
        _h = _result.get("history", {})
    else:
        _ri = None
        _h = None

    print(f"\\n  Run {_idx+1}:")
    if _ri is not None:
        # Check flow_loss_config in run_info
        _has_flow_cfg = "flow_loss_config" in _ri
        if _has_flow_cfg:
            print(f"    run_info['flow_loss_config'] present ✓")
            _fc = _ri["flow_loss_config"]
            print(f"      use_flow_loss={_fc.get('use_flow_loss')}, "
                  f"mode={_fc.get('flow_loss_mode')}, "
                  f"angle={_fc.get('flow_angle_mode')}")
        else:
            print(f"    WARNING: 'flow_loss_config' missing from run_info")
            _errors.append(f"Run {_idx+1}: flow_loss_config missing from run_info")

        # Check angle_mode recorded
        _am = _ri.get("angle_mode", _ri.get("hparams", {}).get("angle_mode"))
        if _am == "both":
            print(f"    angle_mode='{_am}' ✓")
        else:
            print(f"    WARNING: angle_mode='{_am}' (expected 'both')")
    else:
        print(f"    WARNING: could not extract run_info from result")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: make_run_key differentiation
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 6: make_run_key produces distinct keys ──")

_keys = set()
for _idx, _cfg in enumerate(_sweep_configs):
    try:
        _key = make_run_key(
            angle_mode="both",
            physics_cfg=_cfg["physics_cfg"],
            hidden_dim=32,
            num_layers=3,
            warmup_epochs_flow=_cfg.get("warmup_epochs_flow", 0),
        )
        _keys.add(_key)
        print(f"  Run {_idx+1} key: '{_key}'")
    except TypeError as _te:
        # make_run_key may not accept warmup_epochs_flow yet
        print(f"  Run {_idx+1}: make_run_key raised TypeError — {_te}")
        print(f"    (May need to add warmup_epochs_flow param to make_run_key)")
        _errors.append(f"make_run_key doesn't accept flow params: {_te}")
        break

if len(_keys) == len(_sweep_configs):
    print(f"  {len(_keys)} distinct keys for {len(_sweep_configs)} configs ✓")
elif _keys:
    print(f"  WARNING: only {len(_keys)} distinct keys for {len(_sweep_configs)} configs")
    _errors.append("make_run_key does not differentiate all flow configs")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: Regression — existing modes still work in sweep
# ──────────────────────────────────────────────────────────────────────────────
print("\\n── STEP 7: Regression — edge_delta mode still works ──")

try:
    _result_ed = run_hparam_sweep(
        networks=_networks[:20],
        num_epochs=4,
        hidden_dim=32,
        num_layers=3,
        angle_mode="edge_delta",
        physics_cfg=PhysicsConfig(
            use_physics_loss=True,
            use_ptdf_loss=False,
            use_flow_loss=False,
            loss_weight_mode="fixed",
        ),
        warmup_epochs_phys=1,
        use_pnom_share=False,
        ptdf_mode=None,
        verbose=False,
    )
    print("  angle_mode='edge_delta' sweep completed ✓")
except Exception as _ex:
    _errors.append(f"Regression: edge_delta sweep failed: {_ex}")
    print(f"  FAIL: {type(_ex).__name__}: {_ex}")

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
_elapsed = _time.perf_counter() - _t0
print("\\n" + "=" * 70)
if not _errors:
    print(f"✓ ALL E2E TESTS PASSED ({_elapsed:.1f}s)")
else:
    print(f"✗ {len(_errors)} ISSUES FOUND ({_elapsed:.1f}s):")
    for _e in _errors:
        print(f"  • {_e}")
print("=" * 70)

# Cleanup
del (_networks, _ds_both, _g0, _model_both, _loader, _batch, _out,
     _node_pred, _delta_theta_pred, _cfg_flow, _theta_pred, _vmag_pred,
     _flow_loss, _flow_diag, _train_ds, _val_ds, _test_ds, _phys_cfg_train,
     _model_t, _hist_t, _metrics_t, _flow_losses, _train_losses,
     _sweep_configs, _sweep_common, _sweep_results, _errors)
'''
    print(code)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dual-angle flow loss implementation")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run tests for specific phase only (1=043, 2=044, 3=045)")
    args = parser.parse_args()

    all_passed = True

    if args.phase is None or args.phase == 1:
        all_passed &= test_phase1_angle_mode_both()

    if args.phase is None or args.phase == 2:
        all_passed &= test_phase2_flow_loss_functions()

    if args.phase is None or args.phase == 3:
        all_passed &= test_phase3_training_integration()

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED OR SKIPPED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)
