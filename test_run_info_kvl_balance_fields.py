"""Regression guards ensuring run_hparam_sweep's run_info dict (and its Analysis-notebook
consumers) surface fraction_kvl / max_w_kvl / enable_balance_dc / balance_dc_clamp_factor
as first-class fields for every future sweep run (task 081).
See research 2026-09-28-s7-kvl-kcl-activation-and-run-df-gap.md, Finding 5.

Group 0: notebook-source guards (run_info dict keys present in canonical + all 10 session
copies; Analysis create_comparison_dataframe / print_sweep_summary / CONFIG_KEYS updated).
Group 1: numeric -- a real (exec'd) PhysicsConfig instance's 4 fields match a replicated
run_info-subset builder mirroring the exact new dict entries, for both non-default values
and class defaults.
"""
import glob
import hashlib
import json
import os
import re

REPO_ROOT = os.path.dirname(__file__)
TRAINING_NB = os.path.join(REPO_ROOT, "GNN_Powerflow_V2.7_Training.ipynb")
ANALYSIS_NB = os.path.join(REPO_ROOT, "GNN_Powerflow_V2.7_Analysis.ipynb")
SESSION_DIR = os.path.join(REPO_ROOT, "run_in_parallell", "S_sessions")

NEW_FIELDS = ["fraction_kvl", "max_w_kvl", "enable_balance_dc", "balance_dc_clamp_factor"]


def _load_cell_source(path, marker):
    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            return src
    raise AssertionError(f"cell containing {marker!r} not found in {path}")


def _extract_function_body(src, func_name):
    start = src.index(f"def {func_name}(")
    rest = src[start:]
    m = re.search(r"\ndef \w+\(", rest[1:])
    return rest[: m.start() + 1] if m else rest


# -- Group 0: notebook-source guards ------------------------------------------
def test_canonical_run_info_has_new_fields():
    src = _load_cell_source(TRAINING_NB, "run_info = {")
    for field in NEW_FIELDS:
        assert f'"{field}": pcfg.{field}' in src, f"run_info missing {field!r}"
    print("TEST 0a ok")


def test_session_copies_run_info_byte_identical_to_canonical():
    canon_src = _load_cell_source(TRAINING_NB, "def run_hparam_sweep(")
    canon_hash = hashlib.sha256(canon_src.encode()).hexdigest()
    session_files = sorted(glob.glob(os.path.join(SESSION_DIR, "*.ipynb")))
    assert len(session_files) == 10, f"expected 10 session copies, found {len(session_files)}"
    for path in session_files:
        src = _load_cell_source(path, "def run_hparam_sweep(")
        h = hashlib.sha256(src.encode()).hexdigest()
        assert h == canon_hash, f"{os.path.basename(path)} run_hparam_sweep cell diverged from canonical"
    print("TEST 0b ok")


def test_analysis_create_comparison_dataframe_has_new_columns():
    src = _load_cell_source(ANALYSIS_NB, "def create_comparison_dataframe(")
    body = _extract_function_body(src, "create_comparison_dataframe")
    for field in NEW_FIELDS:
        assert f'"{field}":' in body, f"create_comparison_dataframe missing column {field!r}"
    print("TEST 0c ok")


def test_analysis_print_sweep_summary_reads_new_fields():
    src = _load_cell_source(ANALYSIS_NB, "def print_sweep_summary(")
    body = _extract_function_body(src, "print_sweep_summary")
    assert "fraction_kvl" in body and "enable_balance_dc" in body
    print("TEST 0d ok")


def test_analysis_config_keys_has_new_fields():
    src = _load_cell_source(ANALYSIS_NB, "CONFIG_KEYS = [")
    for field in NEW_FIELDS:
        assert f'"{field}"' in src, f"CONFIG_KEYS missing {field!r}"
    print("TEST 0e ok")


# -- Group 1: numeric --------------------------------------------------------
def _exec_notebook_up_to(nb_path, marker):
    with open(nb_path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = nb["cells"]
    idx = next(
        i for i, c in enumerate(cells)
        if c.get("cell_type") == "code" and marker in "".join(c.get("source", []))
    )
    ns = {"__name__": "__run_info_field_test__"}
    for cell in cells[: idx + 1]:
        if cell.get("cell_type") != "code":
            continue
        exec(compile("".join(cell.get("source", [])), "<notebook>", "exec"), ns)  # noqa: S102
    return ns


def _build_run_info_subset(pcfg):
    """Mirrors exactly the 4 new run_info dict entries added in task 081."""
    return {
        "fraction_kvl": pcfg.fraction_kvl,
        "max_w_kvl": pcfg.max_w_kvl,
        "enable_balance_dc": pcfg.enable_balance_dc,
        "balance_dc_clamp_factor": pcfg.balance_dc_clamp_factor,
    }


def test_run_info_subset_matches_physics_config_values():
    ns = _exec_notebook_up_to(TRAINING_NB, "class PhysicsConfig:")
    PhysicsConfig = ns["PhysicsConfig"]
    pcfg = PhysicsConfig(fraction_kvl=0.1, max_w_kvl=42.0,
                          enable_balance_dc=True, balance_dc_clamp_factor=2.5)
    assert _build_run_info_subset(pcfg) == {
        "fraction_kvl": 0.1,
        "max_w_kvl": 42.0,
        "enable_balance_dc": True,
        "balance_dc_clamp_factor": 2.5,
    }
    print("TEST 1a ok")


def test_run_info_subset_defaults():
    ns = _exec_notebook_up_to(TRAINING_NB, "class PhysicsConfig:")
    PhysicsConfig = ns["PhysicsConfig"]
    pcfg = PhysicsConfig()
    assert _build_run_info_subset(pcfg) == {
        "fraction_kvl": 0.0,
        "max_w_kvl": 100.0,
        "enable_balance_dc": False,
        "balance_dc_clamp_factor": 3.0,
    }
    print("TEST 1b ok")


if __name__ == "__main__":
    test_canonical_run_info_has_new_fields()
    test_session_copies_run_info_byte_identical_to_canonical()
    test_analysis_create_comparison_dataframe_has_new_columns()
    test_analysis_print_sweep_summary_reads_new_fields()
    test_analysis_config_keys_has_new_fields()
    test_run_info_subset_matches_physics_config_values()
    test_run_info_subset_defaults()
    print("ALL TESTS PASSED")
