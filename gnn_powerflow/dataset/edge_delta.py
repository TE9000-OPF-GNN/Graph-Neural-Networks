"""BFS-based edge angle utilities for edge_delta angle mode.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 10 /
        GNN_Powerflow_V2.7_Analysis.ipynb (PowerFlowGNN cell)
TODO: remove duplicate inline definitions from notebooks once refactored.
"""
# Source: GNN_Powerflow_V2.7_Training.ipynb cell 10
# Source: GNN_Powerflow_V2.7_Analysis.ipynb (PowerFlowGNN cell)
# TODO: remove duplicate inline definitions from notebooks once refactored.
from __future__ import annotations

from collections import deque

import numpy as np
import torch


def precompute_bfs_order(
    edge_index_fwd: torch.Tensor,
    slack_idx: int,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute BFS traversal from slack for θ reconstruction from Δθ.

    Returns:
        bfs_edge_idx:   [N-1] indices into forward edges — edges in BFS tree order
        bfs_signs:      [N-1] float tensor — +1 or -1
        bfs_node_order: [N-1] node indices in BFS visit order (excludes slack)
        bfs_parent:     [N-1] parent node index for each BFS child

    Convention: θ_child = θ_parent - sign * Δθ_edge
        sign = +1 when edge is parent→child (forward direction = from→to)
        sign = -1 when edge is child→parent (traverse against forward direction)
    """
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
    n_edges = edge_index_fwd.size(1)
    for e in range(n_edges):
        u = edge_index_fwd[0, e].item()
        v = edge_index_fwd[1, e].item()
        adj[u].append((v, e, +1.0))
        adj[v].append((u, e, -1.0))

    visited = [False] * num_nodes
    visited[slack_idx] = True
    queue = deque([slack_idx])

    bfs_edge_idx: list[int] = []
    bfs_signs: list[float] = []
    bfs_node_order: list[int] = []
    bfs_parent: list[int] = []

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
        f"BFS did not reach all nodes: visited {len(bfs_node_order) + 1}/{num_nodes}"
    )

    return (
        torch.tensor(bfs_edge_idx, dtype=torch.long),
        torch.tensor(bfs_signs, dtype=torch.float32),
        torch.tensor(bfs_node_order, dtype=torch.long),
        torch.tensor(bfs_parent, dtype=torch.long),
    )


def reconstruct_theta_from_delta(
    delta_theta_fwd: torch.Tensor,
    bfs_edge_idx: torch.Tensor,
    bfs_signs: torch.Tensor,
    bfs_node_order: torch.Tensor,
    bfs_parent: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
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


def compute_bfs_depth(
    edge_index_fwd: np.ndarray,
    slack_idx: int,
    num_nodes: int,
) -> np.ndarray:
    """
    Compute BFS hop distance from slack bus for each node.
    Returns np.ndarray [num_nodes] with hop distances (slack=0, unreachable=-1).
    """
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    E = edge_index_fwd.shape[1]
    for e in range(E):
        u, v = int(edge_index_fwd[0, e]), int(edge_index_fwd[1, e])
        adj[u].append(v)
        adj[v].append(u)

    depth = np.full(num_nodes, -1, dtype=int)
    depth[slack_idx] = 0
    queue = deque([slack_idx])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if depth[nbr] == -1:
                depth[nbr] = depth[node] + 1
                queue.append(nbr)
    return depth
