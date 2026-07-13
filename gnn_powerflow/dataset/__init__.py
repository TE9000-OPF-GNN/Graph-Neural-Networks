"""Public re-exports for gnn_powerflow.dataset subpackage."""
from gnn_powerflow.dataset.dataset import (
    EDGE_FEATURE_DIM,
    PowerFlowDataset,
    _create_graph_data,
    _build_graph_edges,
    collate_with_ptdf,
    validate_training_data,
    scenario_to_data,
)
from gnn_powerflow.dataset.ptdf import (
    compute_ptdf_matrix,
    compute_admittance_matrix,
    get_pypsa_Y_numpy,
    precompute_Y_matrices,
)
from gnn_powerflow.dataset.edge_delta import (
    precompute_bfs_order,
    reconstruct_theta_from_delta,
    compute_bfs_depth,
)
from gnn_powerflow.dataset.io import (
    save_dataset_list,
    load_dataset_list,
)

__all__ = [
    "EDGE_FEATURE_DIM",
    "PowerFlowDataset",
    "_create_graph_data",
    "_build_graph_edges",
    "collate_with_ptdf",
    "validate_training_data",
    "scenario_to_data",
    "compute_ptdf_matrix",
    "compute_admittance_matrix",
    "get_pypsa_Y_numpy",
    "precompute_Y_matrices",
    "precompute_bfs_order",
    "reconstruct_theta_from_delta",
    "compute_bfs_depth",
    "save_dataset_list",
    "load_dataset_list",
]
