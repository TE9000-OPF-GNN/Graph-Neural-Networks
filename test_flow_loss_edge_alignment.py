"""Regression guards for V2.7 flow-loss edge selection + DC mask alignment.
Protects against two recurring bugs (research 2026-06-30):
  1. [:2] vs [::2] forward-edge selection (MANDATORY RULE 7).
  2. B3 - DC positional [:len(y)] fallback misaligns batched multi-graph data
     with transformers; the boolean dc mask must be used instead.
Plus the edge_delta theta-wiring guard (globals must not detach).
"""
import json, os, torch
import torch.nn.functional as F

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")


def _load_train_cell_source():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "def train_power_flow_gnn" in src:   # scope to the call-site cell only
            return src
    raise AssertionError("train_power_flow_gnn cell not found")


# -- Group 0: notebook-source guard (call site) -------------------------------
def test_notebook_uses_correct_convention():
    src = _load_train_cell_source()
    # the train cell's forward/DC convention (edge_index[:, ::2] lives in the
    # helper cell, NOT here - the train gate uses these two masks):
    assert "forward_edge_mask[::2]" in src, "AC line_mask must be forward_edge_mask[::2]"
    assert "dc_flow_mask[::2]" in src, "DC mask must be dc_flow_mask[::2] (boolean, lines-only)"
    print("TEST 0a ok")


def test_notebook_lacks_regression_markers():
    src = _load_train_cell_source()
    for bad in ("_dc_mask_use = None", "_dc_mask_use=None",
                "edge_index[:, :2]", "edge_attr[:2]"):
        assert bad not in src, f"regression marker reintroduced: {bad!r}"
    print("TEST 0b ok")


def test_notebook_wires_edge_delta_theta():
    src = _load_train_cell_source()
    assert "_resolve_theta_for_global_losses" in src, \
        "edge_delta global losses require theta reconstruction wiring"
    # the flow gate must NOT detach the reconstructed theta (else globals cannot
    # train the delta_theta head). Match the ACTUAL reconstruct call (not the
    # explanatory comment) so flipping the gate to detach=True fails this test.
    assert "device, detach=False)" in src, \
        "flow gate must call _reconstruct_batch_theta(..., detach=False)"
    assert "device, detach=True)" not in src, \
        "flow gate must not detach reconstructed theta (globals could not train it)"
    print("TEST 0c ok")


# -- Replicated helpers (must match the notebook DC sub-loss exactly) ----------
def compute_flow_loss_dc_local(delta_theta_pred, edge_attr_fwd, y_line_p, line_mask=None):
    x_react = edge_attr_fwd[:, 1]
    f_pred_dc = delta_theta_pred / x_react.clamp(min=1e-8)
    f_pred_dc = f_pred_dc[line_mask] if line_mask is not None else f_pred_dc[:len(y_line_p)]
    return F.mse_loss(f_pred_dc, y_line_p), (f_pred_dc - y_line_p).abs().mean().detach()


def build_interleaved_edges(num_lines):
    src = torch.empty(2 * num_lines, dtype=torch.long)
    dst = torch.empty(2 * num_lines, dtype=torch.long)
    fwd = torch.arange(num_lines)
    src[0::2], dst[0::2] = fwd, fwd + 1        # forward at even indices
    src[1::2], dst[1::2] = fwd + 1, fwd        # reverse at odd indices
    edge_index = torch.stack([src, dst], 0)
    fwd_mask = torch.zeros(2 * num_lines, dtype=torch.bool); fwd_mask[0::2] = True
    return edge_index, fwd_mask


# -- Group 1: forward selection [::2] == mask, [:2] is wrong -------------------
def test_stride_equals_forward_mask():
    ei, m = build_interleaved_edges(12)
    assert torch.equal(ei[:, ::2], ei[:, m]) and ei[:, ::2].size(1) == 12
    print("TEST 1a ok")


def test_first_two_slice_is_wrong():
    ei, m = build_interleaved_edges(12)
    assert ei[:, :2].size(1) == 2 != int(m.sum())
    crashed = False
    try:
        _ = torch.randn(12)[m[:2]]             # length-2 boolean on E_fwd targets
    except IndexError:
        crashed = True
    assert crashed, "[:2] boolean mask must raise on length mismatch"
    print("TEST 1b ok")


# -- Group 2: batched-with-transformers DC alignment (B3 guard) ----------------
def _make_graph_fwd(n_lines, n_trafos):
    n_fwd = n_lines + n_trafos
    edge_attr = torch.zeros(n_fwd, 6)
    edge_attr[:, 1] = torch.linspace(0.05, 0.4, n_fwd)     # reactance col 1
    theta = torch.linspace(0.0, 1.0, n_fwd + 1)
    src, dst = torch.arange(n_fwd), torch.arange(1, n_fwd + 1)
    delta = theta[src] - theta[dst]
    dc_mask = torch.zeros(n_fwd, dtype=torch.bool); dc_mask[:n_lines] = True
    f_true = (delta / edge_attr[:, 1])[dc_mask]
    return edge_attr, delta, dc_mask, f_true


def _batch_two_graphs():
    ea0, d0, m0, f0 = _make_graph_fwd(5, 2)
    ea1, d1, m1, f1 = _make_graph_fwd(4, 3)
    return (torch.cat([ea0, ea1]), torch.cat([d0, d1]),
            torch.cat([m0, m1]), torch.cat([f0, f1]))


def test_batched_dc_boolean_vs_positional():
    edge_attr, delta, dc_mask, y_dc = _batch_two_graphs()
    _, mae_bool = compute_flow_loss_dc_local(delta, edge_attr, y_dc, line_mask=dc_mask)
    _, mae_pos = compute_flow_loss_dc_local(delta, edge_attr, y_dc, line_mask=None)
    assert torch.isclose(mae_bool, torch.tensor(0.0), atol=1e-6), "boolean must align"
    assert mae_pos > 100 * mae_bool + 1e-4, "positional fallback must misalign (B3)"
    print(f"TEST 2 ok  bool={mae_bool.item():.2e} pos={mae_pos.item():.2e}")


# -- Group 3: theta must stay differentiable through the DC-global loss (I-1) --
def test_dc_global_keeps_gradient():
    # Mirrors _reconstruct_batch_theta(detach=False): a differentiable map from
    # delta_theta_pred -> theta -> dc_global loss must yield a non-None grad.
    edge_attr, delta, dc_mask, y_dc = _batch_two_graphs()
    dtp = delta.clone().requires_grad_(True)
    # trivial differentiable theta surrogate (cumsum) standing in for BFS recon
    theta = torch.cumsum(dtp, dim=0)
    f_pred = (theta[:-1] - theta[1:])                 # any differentiable use
    loss = F.mse_loss(f_pred[dc_mask[:-1]], y_dc[:f_pred[dc_mask[:-1]].numel()])
    loss.backward()
    assert dtp.grad is not None and torch.isfinite(dtp.grad).all(), \
        "global-flow loss must backprop into delta_theta_pred (no detach at the gate)"
    print("TEST 3 ok")


if __name__ == "__main__":
    test_notebook_uses_correct_convention()
    test_notebook_lacks_regression_markers()
    test_notebook_wires_edge_delta_theta()
    test_stride_equals_forward_mask()
    test_first_two_slice_is_wrong()
    test_batched_dc_boolean_vs_positional()
    test_dc_global_keeps_gradient()
    print("\nAll flow-loss edge-alignment regression tests passed.")
