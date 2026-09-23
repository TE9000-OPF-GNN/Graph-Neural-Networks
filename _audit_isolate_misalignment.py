"""Throwaway audit script #4: isolate the both-mode P/Q column-misalignment bug on a
SHUNT-FREE network (ieee9-like, idx=1) to decouple it from the separately-known
shunt/b_diag omission bug (physics_audit_2026_07_18)."""
import json, os
import torch

NOTEBOOK = os.path.join(os.path.dirname(__file__), "GNN_Powerflow_V2.7_Training.ipynb")
DATA_ROOT = r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN"
DATASET_INDEX = os.path.join(DATA_ROOT, "training_networks_saved", "mixed_1500_w.json")


def load_notebook_cells():
    with open(NOTEBOOK, "r", encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def exec_notebook_upto(marker, ns=None):
    cells = load_notebook_cells()
    ns = ns if ns is not None else {"__name__": "__audit__"}
    idx_target = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code" and marker in "".join(c.get("source", [])):
            idx_target = i
            break
    for cell in cells[: idx_target + 1]:
        if cell.get("cell_type") != "code":
            continue
        exec(compile("".join(cell.get("source", [])), "<notebook>", "exec"), ns)  # noqa: S102
    return ns


ns = exec_notebook_upto("def train_power_flow_gnn(")
import pypsa  # noqa: E402

PowerFlowDataset = ns["PowerFlowDataset"]
compute_physics_residual_edge_local = ns["compute_physics_residual_edge_local"]

with open(DATASET_INDEX, "r") as fh:
    meta = json.load(fh)
save_dir = os.path.dirname(DATASET_INDEX)
files = meta["files"]

net = pypsa.Network(os.path.join(save_dir, files[1]))  # ieee9-like, 0 shunts, 3 trafos
print(f"network: buses={len(net.buses)} shunts={len(net.shunt_impedances)} trafos={len(net.transformers)}")
ds = PowerFlowDataset([net], angle_mode="both")
data = ds.get(0)

y = data.y
node_pred_3col = torch.stack([y[:, 0], y[:, 2], y[:, 3]], dim=1)
node_pred_4col = y.clone()
bus_masks = (data.slack_mask, data.pv_mask, data.pq_mask)

phys_c, p_res_c, q_res_c = compute_physics_residual_edge_local(
    data.y_delta_theta, y[:, 0], node_pred_3col, data.edge_index, data.edge_attr,
    data.g_diag, data.b_diag, data.x, bus_masks=bus_masks)
phys_b, p_res_b, q_res_b = compute_physics_residual_edge_local(
    data.y_delta_theta, y[:, 0], node_pred_4col, data.edge_index, data.edge_attr,
    data.g_diag, data.b_diag, data.x, bus_masks=bus_masks)

print(f"correct (3-col node_pred): p_res={p_res_c.item():.4e} q_res={q_res_c.item():.4e} physics_loss={phys_c.item():.4e}")
print(f"buggy   (4-col node_pred, as 'both' mode passes): p_res={p_res_b.item():.4e} q_res={q_res_b.item():.4e} physics_loss={phys_b.item():.4e}")
print("-> confirms misalignment bug fires independent of the shunt bug (this network has 0 shunts).")
