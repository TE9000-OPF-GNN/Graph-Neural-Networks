"""
Functional tests for Tasks 043/044 (dual-angle head + DC/AC flow loss).

Self-contained: loads the four ``compute_flow_loss_*`` functions directly from
``GNN_Powerflow_V2.7_Training.ipynb`` (code cell 9) and asserts the dual-head
architecture statically from the model cell. Does NOT import the stale,
gitignored ``code_base.py`` (see plan 058, Step 7a).

Usage:
    python test_dual_angle_flow_loss.py            # all phases
    python test_dual_angle_flow_loss.py --phase 1  # architecture assertion only
    python test_dual_angle_flow_loss.py --phase 2  # loss-function functional tests
    pytest test_dual_angle_flow_loss.py -q         # pytest gate
"""
import sys
import json
import argparse
import torch
import torch.nn.functional as F

NOTEBOOK = "GNN_Powerflow_V2.7_Training.ipynb"


# ═══════════════════════════════════════════════════════════════════════════════
# Loaders — pull the CURRENT implementation from the notebook (no code_base)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_notebook_code():
    """Return the list of code-cell source strings from the training notebook."""
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _load_flow_loss_fns():
    """Exec the cell defining the four compute_flow_loss_* functions into a
    namespace with torch/F preloaded, and return that namespace. A missing
    symbol is a real failure (no code_base fallback)."""
    code = _load_notebook_code()
    src = next(s for s in code if "def compute_flow_loss_dc_local(" in s)  # cell 9
    ns = {"torch": torch, "F": F}
    exec(src, ns)
    for name in (
        "compute_flow_loss_dc_local",
        "compute_flow_loss_dc_global",
        "compute_flow_loss_ac_local",
        "compute_flow_loss_ac_global",
    ):
        assert name in ns, f"FAIL: {name} not defined in notebook loss cell"
    return ns


def _model_cell_source():
    """Return the source of the model cell (the one defining the dual heads)."""
    code = _load_notebook_code()
    return next(s for s in code if "self.vang_pred" in s and "self.edge_angle_pred" in s)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Task 043 — angle_mode dual-head architecture (static source assertion)
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase1_angle_mode_both():
    """Assert the dual-head architecture is wired in the model cell, preserving
    the intent that node/edge_delta/both modes all exist. Static check only —
    gradient-to-both-heads is covered on real code by the Step 7 wiring smoke."""
    print("=" * 70)
    print("PHASE 1: dual-head architecture (static source assertion)")
    print("=" * 70)

    src = _model_cell_source()

    assert "self.vang_pred = nn.Linear(" in src, \
        "FAIL: node voltage-angle head (vang_pred) not found"
    assert "self.edge_angle_pred = nn.Linear(" in src, \
        "FAIL: edge angle head (edge_angle_pred) not found"
    assert 'angle_mode == "edge_delta"' in src, \
        "FAIL: edge_delta branch (node_pred [N,3]) not found"
    assert "num_nodes, 4" in src, \
        "FAIL: [num_nodes, 4] assembly for node/both modes not found"
    assert 'angle_mode in ("edge_delta", "both")' in src, \
        "FAIL: edge-angle head is not gated by ('edge_delta', 'both')"

    print("  vang_pred head ............................. present OK")
    print("  edge_angle_pred head ....................... present OK")
    print("  edge_delta branch (node_pred [N,3]) ........ present OK")
    print("  [num_nodes, 4] assembly (node/both) ........ present OK")
    print("  edge-angle head gated by edge_delta/both ... present OK")
    print("\n  Phase 1: architecture assertions PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Task 044 — DC/AC per-edge flow loss functions (functional)
# ═══════════════════════════════════════════════════════════════════════════════

def test_phase2_flow_loss_functions():
    """Validate the four flow-loss variants against the CURRENT notebook
    signatures: forward-sliced edge attrs/indices, positional y_line_q for AC,
    grad flow to Δθ (local), θ (global), and V (AC), plus DC local==global
    consistency."""
    print("\n" + "=" * 70)
    print("PHASE 2: DC/AC per-edge flow loss functions")
    print("=" * 70)

    ns = _load_flow_loss_fns()
    compute_flow_loss_dc_local = ns["compute_flow_loss_dc_local"]
    compute_flow_loss_dc_global = ns["compute_flow_loss_dc_global"]
    compute_flow_loss_ac_local = ns["compute_flow_loss_ac_local"]
    compute_flow_loss_ac_global = ns["compute_flow_loss_ac_global"]

    device = torch.device("cpu")
    E_fwd = 20
    N = 14
    E_supervised = 15  # fewer than E_fwd (lines only, no transformers)

    # ── Synthetic data ──
    delta_theta_pred = (torch.randn(E_fwd, device=device) * 0.05).requires_grad_(True)
    theta_pred = (torch.randn(N, device=device) * 0.05).requires_grad_(True)
    vmag = (torch.ones(N, device=device) + torch.randn(N) * 0.02).requires_grad_(True)

    # Edge attr: [r, x, b_half, tap, g_ser, b_ser] — interleaved fwd/bwd
    edge_attr = torch.zeros(E_fwd * 2, 6, device=device)
    x_react = torch.rand(E_fwd) * 0.1 + 0.01  # reactance 0.01–0.11
    r_resist = x_react * 0.1                   # r/x ≈ 0.1
    g_ser = r_resist / (r_resist ** 2 + x_react ** 2)
    b_ser = -x_react / (r_resist ** 2 + x_react ** 2)
    for i in range(E_fwd):
        for row in (2 * i, 2 * i + 1):  # forward + reverse share series params
            edge_attr[row, 0] = r_resist[i]
            edge_attr[row, 1] = x_react[i]
            edge_attr[row, 4] = g_ser[i]
            edge_attr[row, 5] = b_ser[i]

    # Edge index (interleaved fwd/bwd)
    src = torch.randint(0, N, (E_fwd,))
    dst = torch.randint(0, N, (E_fwd,))
    edge_index = torch.zeros(2, E_fwd * 2, dtype=torch.long)
    edge_index[0, ::2] = src
    edge_index[1, ::2] = dst
    edge_index[0, 1::2] = dst  # reverse
    edge_index[1, 1::2] = src

    # Forward-sliced views expected by the CURRENT signatures
    edge_attr_fwd = edge_attr[::2]        # [E_fwd, 6]
    edge_index_fwd = edge_index[:, ::2]   # [2, E_fwd]

    # True flows: DC approximation of our angles (so loss should be small)
    y_line_p = delta_theta_pred[:E_supervised].detach() / x_react[:E_supervised]
    y_line_p = y_line_p + torch.randn(E_supervised) * 0.001
    y_line_q = torch.randn(E_supervised) * 0.1

    # ── 2a: DC local — scalar loss with grad to Δθ ──
    print("\n  [2a] compute_flow_loss_dc_local")
    loss_dc_l, mae_dc_l = compute_flow_loss_dc_local(
        delta_theta_pred, edge_attr_fwd, y_line_p)
    assert loss_dc_l.requires_grad, "FAIL: DC-local loss has no grad_fn"
    assert loss_dc_l.item() >= 0, "FAIL: negative DC-local loss"
    assert mae_dc_l.item() >= 0, "FAIL: negative DC-local MAE"
    loss_dc_l.backward(retain_graph=True)
    assert delta_theta_pred.grad is not None, "FAIL: no grad on delta_theta (DC-local)"
    print(f"    loss={loss_dc_l.item():.6f}, mae={mae_dc_l.item():.6f} OK")
    delta_theta_pred.grad = None

    # ── 2b: DC global — scalar loss with grad to θ ──
    print("\n  [2b] compute_flow_loss_dc_global")
    loss_dc_g, mae_dc_g = compute_flow_loss_dc_global(
        theta_pred, edge_index_fwd, edge_attr_fwd, y_line_p)
    assert loss_dc_g.requires_grad, "FAIL: DC-global loss has no grad_fn"
    loss_dc_g.backward(retain_graph=True)
    assert theta_pred.grad is not None, "FAIL: no grad on theta_pred (DC-global)"
    print(f"    loss={loss_dc_g.item():.6f}, mae={mae_dc_g.item():.6f} OK")
    theta_pred.grad = None

    # ── 2c: Consistency — DC local == DC global when Δθ = θ_src − θ_dst ──
    print("\n  [2c] Consistency: DC local == DC global (Δθ = θ_src − θ_dst)")
    with torch.no_grad():
        consistent_delta = theta_pred[edge_index_fwd[0]] - theta_pred[edge_index_fwd[1]]
    consistent_delta_param = consistent_delta.clone().requires_grad_(True)
    y_line_p_consist = torch.zeros(E_supervised)  # dummy target
    loss_l, _ = compute_flow_loss_dc_local(
        consistent_delta_param, edge_attr_fwd, y_line_p_consist)
    loss_g, _ = compute_flow_loss_dc_global(
        theta_pred, edge_index_fwd, edge_attr_fwd, y_line_p_consist)
    diff = abs(loss_l.item() - loss_g.item())
    assert diff < 1e-5, \
        f"FAIL: local={loss_l.item():.8f} vs global={loss_g.item():.8f}, diff={diff:.2e}"
    print(f"    DC local={loss_l.item():.8f}, DC global={loss_g.item():.8f}, diff={diff:.2e} OK")

    # ── 2d: AC local — grad to Δθ AND V ──
    print("\n  [2d] compute_flow_loss_ac_local — grad to Δθ and V")
    loss_ac_l, diag_ac_l = compute_flow_loss_ac_local(
        delta_theta_pred, vmag, edge_index_fwd, edge_attr_fwd, y_line_p, y_line_q)
    assert loss_ac_l.requires_grad, "FAIL: AC-local loss has no grad_fn"
    loss_ac_l.backward(retain_graph=True)
    assert delta_theta_pred.grad is not None, "FAIL: no grad on delta_theta (AC-local)"
    assert vmag.grad is not None, "FAIL: no grad on vmag (AC-local must use V)"
    print(f"    loss={loss_ac_l.item():.6f}, p_mae={diag_ac_l['p_mae']:.6f}, "
          f"q_mae={diag_ac_l['q_mae']:.6f} OK (vmag grad norm={vmag.grad.norm():.6f})")
    delta_theta_pred.grad = None
    vmag.grad = None

    # ── 2e: AC global — grad to θ AND V ──
    print("\n  [2e] compute_flow_loss_ac_global — grad to θ and V")
    loss_ac_g, diag_ac_g = compute_flow_loss_ac_global(
        theta_pred, vmag, edge_index_fwd, edge_attr_fwd, y_line_p, y_line_q)
    assert loss_ac_g.requires_grad, "FAIL: AC-global loss has no grad_fn"
    loss_ac_g.backward(retain_graph=True)
    assert theta_pred.grad is not None, "FAIL: no grad on theta_pred (AC-global)"
    assert vmag.grad is not None, "FAIL: no grad on vmag (AC-global)"
    print(f"    loss={loss_ac_g.item():.6f}, theta grad norm={theta_pred.grad.norm():.6f}, "
          f"V grad norm={vmag.grad.norm():.6f} OK")
    theta_pred.grad = None
    vmag.grad = None

    # ── 2f: AC P+Q flow supervision (flow_target='pq') ──
    print("\n  [2f] AC flow loss — P+Q supervision")
    loss_pq, diag_pq = compute_flow_loss_ac_local(
        delta_theta_pred, vmag, edge_index_fwd, edge_attr_fwd,
        y_line_p, y_line_q, flow_target="pq")
    assert loss_pq.item() > 0, "FAIL: P+Q loss should be > 0"
    assert "p_mae" in diag_pq and "q_mae" in diag_pq, \
        "FAIL: AC diagnostics missing p_mae/q_mae"
    print(f"    AC local (P+Q): loss={loss_pq.item():.6f}, "
          f"p_mae={diag_pq['p_mae']:.6f}, q_mae={diag_pq['q_mae']:.6f} OK")

    print("\n  Phase 2: 2a–2f PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dual-angle flow loss implementation")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run tests for specific phase only (1=architecture, 2=loss fns)")
    args = parser.parse_args()

    all_passed = True
    for phase, fn in ((1, test_phase1_angle_mode_both), (2, test_phase2_flow_loss_functions)):
        if args.phase is None or args.phase == phase:
            try:
                fn()
            except AssertionError as exc:
                all_passed = False
                print(f"  x Phase {phase} FAILED: {exc}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)
