import pickle

from migrate_kvl_activation_backfill import migrate_file, _regenerate_run_key


def _make_old_run(**overrides):
    base = {
        "tag": "r1", "physics_label": "phys0.1_ptdf0.1",
        "batch_size": 64, "lr": 1e-3, "hidden_dim": 64, "num_layers": 3,
        "conv_type": "gcn", "y_matrix_source": "pypsa", "seed": 42,
        "head_mode": "standard", "use_pnom_share": False,
        "use_residual": False, "norm_type": None, "activation": "leaky_relu",
        "use_dropout": False, "drop_rate": 0.1,
        "angle_mode": "node", "vmag_mode": "absolute", "use_global_pool": False,
        "physics_activation_start": 10, "physics_activation_stop": None,
        "ptdf_activation_start": 10, "ptdf_activation_stop": None,
        "activation_ramp_epochs": 5,
        "run_key": "STALE_KEY_WITHOUT_KVL_TOKEN",
    }
    base.update(overrides)
    return base


def test_migrate_adds_default_kvl_activation_fields(tmp_path):
    p = tmp_path / "old_runs.json"
    pickle.dump([_make_old_run()], open(p, "wb"))
    migrate_file(str(p))
    migrated = pickle.load(open(p, "rb"))
    assert migrated[0]["kvl_activation_start"] == 0
    assert migrated[0]["kvl_activation_stop"] is None


def test_migrate_regenerates_run_key_with_kvl_token(tmp_path):
    p = tmp_path / "old_runs.json"
    pickle.dump([_make_old_run()], open(p, "wb"))
    migrate_file(str(p))
    migrated = pickle.load(open(p, "rb"))
    assert "_kvl0_" in migrated[0]["run_key"] or migrated[0]["run_key"].endswith("_kvl0")


def test_migrate_skips_runs_already_having_kvl_fields(tmp_path):
    p = tmp_path / "new_runs.json"
    r = _make_old_run(kvl_activation_start=10, kvl_activation_stop=60,
                       run_key="ALREADY_CURRENT_kvl10-60")
    pickle.dump([r], open(p, "wb"))
    migrate_file(str(p))
    migrated = pickle.load(open(p, "rb"))
    assert migrated[0]["run_key"] == "ALREADY_CURRENT_kvl10-60"  # untouched


def test_migrate_leaves_runs_without_run_key_untouched(tmp_path):
    p = tmp_path / "ancient_runs.json"
    r = _make_old_run()
    del r["run_key"]
    pickle.dump([r], open(p, "wb"))
    migrate_file(str(p))
    migrated = pickle.load(open(p, "rb"))
    assert "run_key" not in migrated[0]
    assert migrated[0]["kvl_activation_start"] == 0


def test_migrate_creates_backup_file(tmp_path):
    import os
    p = tmp_path / "old_runs.json"
    pickle.dump([_make_old_run()], open(p, "wb"))
    migrate_file(str(p))
    assert os.path.isfile(str(p) + ".bak_preKvlBackfill")
