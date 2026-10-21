"""Dataset IO — save/load lists of solved PyPSA networks.

Source: GNN_Powerflow_V2.7_Training.ipynb cell 7
TODO: remove duplicate from notebook once refactored.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pypsa


def save_dataset_list(
    dataset_list: list,
    index_path: str,
    tag: str = "dataset",
    subfolder: str | None = None,
) -> str:
    """
    Save a list of PyPSA networks using NetCDF (avoids weakref pickle errors).

    Each network is saved as an individual .nc file inside a dedicated subfolder;
    an index JSON records the ordered list of paths so they can be reloaded in
    the same order.

    Parameters
    ----------
    dataset_list : list[pypsa.Network]
    index_path   : str        — path for the JSON index file
    tag          : str        — prefix for individual network filenames
    subfolder    : str | None — name of the subdirectory inside the index
                                directory that holds the .nc files.
                                Defaults to ``tag`` when not supplied.

    Returns
    -------
    str — path to the JSON index file
    """
    save_dir = os.path.dirname(index_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nc_subdir = subfolder if subfolder is not None else tag
    nc_dir = os.path.join(save_dir, nc_subdir)
    os.makedirs(nc_dir, exist_ok=True)

    nc_paths: list[str] = []
    for i, net in enumerate(dataset_list):
        nc_name = f"{tag}_{timestamp}_{i:04d}.nc"
        nc_path = os.path.join(nc_dir, nc_name)
        net.export_to_netcdf(nc_path)
        nc_paths.append(os.path.join(nc_subdir, nc_name))  # relative to save_dir

    with open(index_path, "w") as f:
        json.dump({"tag": tag, "timestamp": timestamp, "files": nc_paths}, f, indent=2)

    print(f"Saved {len(nc_paths)} networks → {nc_dir}/ (index: {index_path})")
    return index_path


def load_dataset_list(index_path: str) -> list:
    """
    Reload a list of PyPSA networks saved by save_dataset_list.
    """
    with open(index_path, "r") as f:
        meta = json.load(f)

    save_dir = os.path.dirname(index_path) or "."
    networks: list[pypsa.Network] = []
    for nc_name in meta["files"]:
        nc_path = os.path.join(save_dir, nc_name)
        networks.append(pypsa.Network(nc_path))

    print(f"Loaded {len(networks)} networks from {index_path}")
    return networks
