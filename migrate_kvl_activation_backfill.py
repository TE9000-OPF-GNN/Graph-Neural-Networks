"""One-shot migration: backfill kvl_activation_start/stop into old sweep-result
files and regenerate their run_key so run_hparam_sweep's checkpoint-resume logic
recognizes them under today's key format (which now always includes a kvl{...} token).

Usage:
    python migrate_kvl_activation_backfill.py <path1> [<path2> ...]

Each file is backed up to `<path>.bak_preKvlBackfill` before being overwritten in place.
Operates on whatever list-of-dicts a pickle-format file deserializes to, regardless of
its .json or .pkl extension (matches load_sweep_results' own unconditional pickle.load
convention in GNN_Powerflow_V2.7_Analysis.ipynb).
"""
import argparse
import pickle
import shutil

_KVL_ACTIVATION_DEFAULTS = {"kvl_activation_start": 0, "kvl_activation_stop": None}


def _regenerate_run_key(r: dict) -> str:
    """Mirror GNN_Powerflow_V2.7_Training.ipynb's make_run_key() string format,
    driven by run_info fields already recorded on the dict (no live PhysicsConfig/
    pcfg object is available for old runs). Keep in sync with make_run_key() if its
    format changes again."""
    cb_parts = []
    if r.get("use_residual", False):
        cb_parts.append("res")
    norm_type = r.get("norm_type")
    if norm_type is not None:
        cb_parts.append(f"n{norm_type[:3]}")
    activation = r.get("activation", "leaky_relu")
    if activation != "leaky_relu":
        cb_parts.append(f"a{activation[:4]}")
    if r.get("use_dropout", False):
        cb_parts.append(f"do{r.get('drop_rate', 0.1)}")
    conv_block_key = "_".join(cb_parts) if cb_parts else "plain"

    angle_mode = r.get("angle_mode", "node")
    vmag_mode = r.get("vmag_mode", "absolute")
    am_part = f"_am{angle_mode}" if angle_mode != "node" else ""
    vm_part = f"_vm{vmag_mode}" if vmag_mode != "absolute" else ""
    gp_part = "_gp" if r.get("use_global_pool", False) else ""

    act_parts = []

    def _window(prefix, start_key, stop_key):
        start = r.get(start_key)
        if start is None:
            return
        token = f"{prefix}{start}"
        stop = r.get(stop_key)
        if stop is not None:
            token += f"-{stop}"
        act_parts.append(token)

    _window("phys", "physics_activation_start", "physics_activation_stop")
    _window("ptdf", "ptdf_activation_start", "ptdf_activation_stop")
    _window("dcfL", "dcf_local_activation_start", "dcf_local_activation_stop")
    _window("dcfG", "dcf_global_activation_start", "dcf_global_activation_stop")
    _window("acfL", "acf_local_activation_start", "acf_local_activation_stop")
    _window("acfG", "acf_global_activation_start", "acf_global_activation_stop")
    _window("kvl", "kvl_activation_start", "kvl_activation_stop")  # always fires: default start=0

    ramp_ep = r.get("activation_ramp_epochs", 5)
    if ramp_ep != 5:
        act_parts.append(f"ramp{ramp_ep}")
    act_key = "_" + "_".join(act_parts) if act_parts else ""

    return (
        f"{r.get('physics_label', '')}_"
        f"bs{r.get('batch_size')}_"
        f"lr{r.get('lr'):.2e}_"
        f"h{r.get('hidden_dim')}_"
        f"L{r.get('num_layers')}_"
        f"{r.get('conv_type')}_"
        f"{r.get('y_matrix_source')}_"
        f"s{r.get('seed', 42)}_"
        f"hm{r.get('head_mode', 'standard')}_"
        f"pn{int(r.get('use_pnom_share', False))}_"
        f"cb-{conv_block_key}"
        f"{am_part}"
        f"{vm_part}"
        f"{gp_part}"
        f"{act_key}"
    )


def migrate_file(path: str) -> None:
    backup_path = path + ".bak_preKvlBackfill"
    shutil.copy2(path, backup_path)

    with open(path, "rb") as f:
        runs = pickle.load(f)

    n_total = len(runs)
    n_migrated = n_already_ok = n_skipped_no_key = 0

    for r in runs:
        if "kvl_activation_start" in r and "kvl_activation_stop" in r:
            n_already_ok += 1
            continue
        r.update(_KVL_ACTIVATION_DEFAULTS)
        if r.get("run_key"):
            r["run_key"] = _regenerate_run_key(r)
            n_migrated += 1
        else:
            n_skipped_no_key += 1  # no run_key at all -- left to run_hparam_sweep's own
                                    # legacy-backfill branch on next resume

    with open(path, "wb") as f:
        pickle.dump(runs, f)

    print(
        f"{path}: {n_total} runs total, {n_migrated} migrated (kvl keys added + run_key "
        f"regenerated), {n_already_ok} already had kvl_activation fields (untouched), "
        f"{n_skipped_no_key} had no run_key at all (left for next resume). "
        f"Backup saved to {backup_path}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="One or more sweep-result files to migrate")
    args = parser.parse_args(argv)
    for path in args.paths:
        migrate_file(path)


if __name__ == "__main__":
    main()
