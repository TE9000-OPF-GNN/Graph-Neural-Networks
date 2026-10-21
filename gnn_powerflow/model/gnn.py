"""GNN model for power flow prediction.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 13 /
        GNN_Powerflow_V2.7_Analysis.ipynb (PowerFlowGNN cell)
TODO: remove duplicate inline definitions from notebooks once refactored.

Note: this copy uses the same 7-edge-feature input as the new package
PowerFlowDataset. Old pkl models trained with 6 edge features are NOT
compatible with this module — use the original notebook code for those.
"""
# Source: GNN_Powerflow_V2.7_Training.ipynb cell 13
# Source: GNN_Powerflow_V2.7_Analysis.ipynb (PowerFlowGNN cell)
# TODO: remove duplicate inline definitions from notebooks once refactored.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GraphConv, TransformerConv


class PowerFlowGNN(nn.Module):
    """
    Graph Attention Network (GATv2) for power flow prediction.
    Predicts per-bus: [vmag, vang, P, Q]
    Predicts per-edge: Δθ (optional, angle_mode="edge_delta" or "both")
    """

    def __init__(
        self,
        node_features: int = 7,
        edge_features: int = 7,   # default 7 for package version
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        conv_type: str = "gatv2",
        head_mode: str = "standard",
        use_residual: bool = False,
        norm_type: str | None = None,
        activation: str = "leaky_relu",
        use_dropout: bool = False,
        drop_rate: float = 0.1,
        conv_mode: str | None = None,  # deprecated shorthand
        angle_mode: str = "node",
        vmag_mode: str = "absolute",
        use_global_pool: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.conv_type = conv_type
        self.head_mode = head_mode
        self.angle_mode = angle_mode
        self.vmag_mode = vmag_mode
        self.use_global_pool = use_global_pool

        # Backward compat: deprecated conv_mode shorthand
        if conv_mode is not None:
            _map = {
                "old":           (False, None,    "leaky_relu", False),
                "residual":      (True,  None,    "gelu",       False),
                "res_norm":      (True,  "layer", "gelu",       False),
                "res_norm_drop": (True,  "layer", "gelu",       True),
            }
            use_residual, norm_type, activation, use_dropout = _map[conv_mode]

        self.use_residual = use_residual
        self.norm_type = norm_type
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate
        self.activation_name = activation

        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # Conv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, edge_features))

        if self.norm_type is not None:
            if self.norm_type == "layer":
                self.norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
                )
            elif self.norm_type == "graph":
                from torch_geometric.nn.norm import GraphNorm
                self.norms = nn.ModuleList(
                    [GraphNorm(hidden_dim) for _ in range(num_layers)]
                )
            else:
                raise ValueError(f"Unknown norm_type: {self.norm_type!r}")

        if self.use_dropout:
            self.drop = nn.Dropout(self.drop_rate)

        if self.use_global_pool:
            self.global_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        _act_map = {
            "leaky_relu": F.leaky_relu,
            "gelu": F.gelu,
            "relu": F.relu,
            "elu": F.elu,
            "silu": F.silu,
        }
        if activation not in _act_map:
            raise ValueError(f"Unknown activation: {activation!r}")
        self.activation_fn = _act_map[activation]

        # Output heads
        if head_mode == "standard":
            if angle_mode == "edge_delta":
                self.vmag_pred = nn.Linear(hidden_dim, 1)
                self.p_pred = nn.Linear(hidden_dim, 1)
                self.q_pred = nn.Linear(hidden_dim, 1)
            else:
                self.vmag_pred = nn.Linear(hidden_dim, 1)
                self.vang_pred = nn.Linear(hidden_dim, 1)
                self.p_pred = nn.Linear(hidden_dim, 1)
                self.q_pred = nn.Linear(hidden_dim, 1)
        elif head_mode == "with_encoder":
            if angle_mode == "edge_delta":
                self.pq_head = nn.Linear(hidden_dim, 1)
                self.pv_head = nn.Linear(hidden_dim, 1)
                self.slack_head = nn.Linear(hidden_dim, 2)
            else:
                self.pq_head = nn.Linear(hidden_dim, 2)
                self.pv_head = nn.Linear(hidden_dim, 2)
                self.slack_head = nn.Linear(hidden_dim, 2)

        # Edge MLP: (h_src || h_dst || edge_attr) -> hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.LeakyReLU(),
        )

        if angle_mode in ("edge_delta", "both"):
            self.edge_angle_pred = nn.Linear(hidden_dim, 1)

        # Bilinear PTDF
        self.ptdf_W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        self._initialize_weights()

    def _make_conv(self, in_dim: int, out_dim: int, edge_features: int):
        if self.conv_type == "gatv2":
            return GATv2Conv(
                in_dim, out_dim // self.heads,
                heads=self.heads, edge_dim=edge_features, concat=True, dropout=0.0,
            )
        elif self.conv_type == "transformer":
            return TransformerConv(
                in_dim, out_dim // self.heads,
                heads=self.heads, edge_dim=edge_features, concat=True, dropout=0.0,
                beta=not self.use_residual,
            )
        elif self.conv_type == "gcn":
            return GCNConv(in_dim, out_dim, add_self_loops=False)
        elif self.conv_type == "graphconv":
            return GraphConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

    def _apply_conv(self, conv, h, edge_index, edge_attr):
        if isinstance(conv, (GATv2Conv, TransformerConv)):
            return conv(h, edge_index, edge_attr)
        return conv(h, edge_index)

    def forward(self, data, return_embeddings: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.node_embedding(x)

        for i, conv in enumerate(self.convs):
            h_in = h
            h = self._apply_conv(conv, h, edge_index, edge_attr)
            if self.norm_type is not None:
                h = self.norms[i](h)
            h = self.activation_fn(h)
            if self.use_dropout:
                h = self.drop(h)
            if self.use_residual:
                h = h + h_in

        h_nodes = h

        if self.use_global_pool:
            from torch_geometric.nn import global_mean_pool, global_max_pool
            batch_idx = (
                data.batch
                if hasattr(data, "batch") and data.batch is not None
                else torch.zeros(h_nodes.size(0), dtype=torch.long, device=h_nodes.device)
            )
            h_mean = global_mean_pool(h_nodes, batch_idx)
            h_max = global_max_pool(h_nodes, batch_idx)
            h_global = self.global_proj(torch.cat([h_mean, h_max], dim=-1))
            h_nodes = h_nodes + h_global[batch_idx]

        # Node prediction heads
        if self.head_mode == "standard":
            if self.angle_mode == "edge_delta":
                vmag_pred = self.vmag_pred(h_nodes)
                if self.vmag_mode == "residual":
                    vmag_pred = vmag_pred + 1.0
                node_pred = torch.cat([vmag_pred, self.p_pred(h_nodes), self.q_pred(h_nodes)], dim=-1)
            else:
                vmag_pred = self.vmag_pred(h_nodes)
                if self.vmag_mode == "residual":
                    vmag_pred = vmag_pred + 1.0
                node_pred = torch.cat(
                    [vmag_pred, self.vang_pred(h_nodes), self.p_pred(h_nodes), self.q_pred(h_nodes)],
                    dim=-1,
                )
        elif self.head_mode == "with_encoder":
            pq_mask = data.pq_mask.bool()
            pv_mask = data.pv_mask.bool()
            slack_mask = data.slack_mask.bool()
            if self.angle_mode == "edge_delta":
                node_pred = torch.zeros(h_nodes.size(0), 3, device=h_nodes.device)
                if pq_mask.any():
                    raw_vmag = self.pq_head(h_nodes[pq_mask]).squeeze(-1)
                    node_pred[pq_mask, 0] = raw_vmag + 1.0 if self.vmag_mode == "residual" else raw_vmag
                if pv_mask.any():
                    node_pred[pv_mask, 2] = self.pv_head(h_nodes[pv_mask]).squeeze(-1)
                if slack_mask.any():
                    slack_out = self.slack_head(h_nodes[slack_mask])
                    node_pred[slack_mask, 1] = slack_out[:, 0]
                    node_pred[slack_mask, 2] = slack_out[:, 1]
            else:
                node_pred = torch.zeros(h_nodes.size(0), 4, device=h_nodes.device)
                if pq_mask.any():
                    pq_out = self.pq_head(h_nodes[pq_mask])
                    if self.vmag_mode == "residual":
                        pq_out = pq_out.clone(); pq_out[:, 0] += 1.0
                    node_pred[pq_mask, 0:2] = pq_out
                if pv_mask.any():
                    pv_out = self.pv_head(h_nodes[pv_mask])
                    node_pred[pv_mask, 1] = pv_out[:, 0]
                    node_pred[pv_mask, 3] = pv_out[:, 1]
                if slack_mask.any():
                    slack_out = self.slack_head(h_nodes[slack_mask])
                    node_pred[slack_mask, 2] = slack_out[:, 0]
                    node_pred[slack_mask, 3] = slack_out[:, 1]
        else:
            raise ValueError(f"Unknown head_mode: {self.head_mode!r}")

        # Edge embeddings
        row, col = data.edge_index
        h_src = h_nodes[row]
        h_dst = h_nodes[col]
        edge_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        h_edges = self.edge_mlp(edge_input)

        # Edge angle prediction
        delta_theta_pred = None
        if self.angle_mode in ("edge_delta", "both"):
            all_fwd = torch.zeros(h_edges.size(0), dtype=torch.bool, device=h_edges.device)
            all_fwd[::2] = True  # MANDATORY: [::2] not [:2]
            delta_theta_pred = self.edge_angle_pred(h_edges[all_fwd]).squeeze(-1)

        if return_embeddings:
            return node_pred, h_nodes, h_edges, delta_theta_pred

        H_W = h_edges @ self.ptdf_W
        ptdf_pred = H_W @ h_nodes.T
        return node_pred, ptdf_pred, delta_theta_pred

    def _initialize_weights(self, gain: float = 1.0):
        nn.init.xavier_uniform_(self.node_embedding.weight, gain=gain)
        nn.init.zeros_(self.node_embedding.bias)

        for conv in self.convs:
            for attr in ("lin_l", "lin_r", "lin_edge"):
                layer = getattr(conv, attr, None)
                if layer is not None and hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)

        _vmag_bias = 0.0 if self.vmag_mode == "residual" else 1.0
        if self.head_mode == "standard":
            if self.angle_mode == "edge_delta":
                prediction_layers = [
                    (self.vmag_pred, _vmag_bias),
                    (self.p_pred, 0.0),
                    (self.q_pred, 0.0),
                ]
            else:
                prediction_layers = [
                    (self.vmag_pred, _vmag_bias),
                    (self.vang_pred, 0.0),
                    (self.p_pred, 0.0),
                    (self.q_pred, 0.0),
                ]
            for layer, bias_init in prediction_layers:
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, bias_init)
            nn.init.zeros_(self.vmag_pred.weight)
        elif self.head_mode == "with_encoder":
            for layer in (self.pq_head, self.pv_head, self.slack_head):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
            nn.init.constant_(self.pq_head.bias[0], _vmag_bias)
            nn.init.zeros_(self.pq_head.weight[0])

        if self.angle_mode in ("edge_delta", "both"):
            nn.init.zeros_(self.edge_angle_pred.weight)
            nn.init.zeros_(self.edge_angle_pred.bias)

        if self.use_global_pool:
            nn.init.xavier_uniform_(self.global_proj.weight)
