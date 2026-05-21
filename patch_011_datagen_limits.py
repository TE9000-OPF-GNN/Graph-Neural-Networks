"""
Patch script for task 011-BUGFIX-datagen-physical-limits-and-pu-correctness
Changes:
  A) Cell 19: clamp b_val >= 0 in load_system_from_csv
  B) Cell 19: set n.sn_mva = 100.0 after pypsa.Network()
  G) Cell 16: add max_balance_err param + KCL balance check to sanity_check_power_flow
"""
import json, sys

NB = "GNN_Powerflow_V2.6_DataGen.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── helpers ──────────────────────────────────────────────────────────────────

def patch_cell(idx: int, old: str, new: str, label: str) -> bool:
    src = "".join(cells[idx].get("source", []))
    if old not in src:
        print(f"  MISS [{label}]: pattern not found in cell {idx}")
        return False
    count = src.count(old)
    if count > 1:
        print(f"  WARN [{label}]: pattern found {count} times in cell {idx} — patching first")
    patched = src.replace(old, new, 1)
    cells[idx]["source"] = list(line + ("\n" if not line.endswith("\n") else "")
                                 for line in patched.split("\n"))
    # fix last element: splitlines strips trailing \n, so last item gets a spurious \n
    # use a safer approach: rebuild from splitlines preserving original line endings
    cells[idx]["source"] = _to_source_list(patched)
    print(f"  OK  [{label}]")
    return True


def _to_source_list(text: str) -> list:
    """Convert plain string back to notebook source list (lines ending with \\n except last)."""
    lines = text.split("\n")
    result = []
    for i, ln in enumerate(lines):
        if i < len(lines) - 1:
            result.append(ln + "\n")
        else:
            if ln:          # non-empty last line
                result.append(ln)
            # empty last line → don't append (avoids trailing blank entry)
    return result


# ── Patch A: clamp b_val ≥ 0 (cell 19) ──────────────────────────────────────
OLD_A = '        b_val = float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0)'
NEW_A = '        b_val = max(float(row.get("b1", 0.0) or 0.0) + float(row.get("b2", 0.0) or 0.0), 0.0)  # [STSI 210526]: clamp b>=0, defense-in-depth for stale CSVs with negative shunt charging'
patch_cell(19, OLD_A, NEW_A, "A: clamp b>=0")

# ── Patch B: set sn_mva = 100.0 (cell 19) ────────────────────────────────────
OLD_B = '    n = pypsa.Network()\n\n    # Buses'
NEW_B = '    n = pypsa.Network()\n    n.sn_mva = 100.0  # [STSI 210526]: cosmetic: all CSV data is on 100 MVA base\n\n    # Buses'
patch_cell(19, OLD_B, NEW_B, "B: n.sn_mva=100")

# ── Patch G1: add max_balance_err to signature (cell 16) ─────────────────────
OLD_G1 = '    max_angle: float = 180 #[STSI070526]: check for voltage angles within ±180° to catch physically unrealistic solutions\n) -> None:'
NEW_G1 = '    max_angle: float = 180,  # [STSI070526]: check for voltage angles within ±180° to catch physically unrealistic solutions\n    max_balance_err: float = 1e-3,  # [STSI 210526]: KCL balance check threshold\n) -> None:'
patch_cell(16, OLD_G1, NEW_G1, "G1: add max_balance_err param")

# ── Patch G2: insert KCL check before logger.info (cell 16) ──────────────────
KCL_BLOCK = '''\n    # 2c) System power balance (KCL)  [STSI 210526]: catch modeling inconsistencies
    if hasattr(network, "generators_t") and hasattr(network.generators_t, "p"):
        gen_p     = float(network.generators_t.p.values.sum())
        load_p    = float(network.loads_t.p.values.sum())
        line_loss = float((network.lines_t.p0.values + network.lines_t.p1.values).sum())
        traf_arr  = network.transformers_t.p0.values + network.transformers_t.p1.values
        traf_loss = float(traf_arr.sum()) if traf_arr.size > 0 else 0.0
        balance_err = abs(gen_p - load_p - line_loss - traf_loss) / max(abs(gen_p), 1e-6)
        if balance_err > max_balance_err:
            raise RuntimeError(
                f"{base_system}: power balance error {balance_err:.2e} > {max_balance_err:.0e} "
                f"(gen={gen_p:.4f}, load={load_p:.4f}, losses={line_loss+traf_loss:.6f})"
            )

    logger.info(
        f"{base_system}: PF sanity OK "
        f"(buses={len(network.buses)}, gens={len(network.generators)}, "
        f"loads={len(network.loads)})"
    )'''

OLD_G2 = '''\n    logger.info(
        f"{base_system}: PF sanity OK "
        f"(buses={len(network.buses)}, gens={len(network.generators)}, "
        f"loads={len(network.loads)})"
    )'''

patch_cell(16, OLD_G2, KCL_BLOCK, "G2: KCL balance check")

# ── Write back ────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("\nDone. Run git diff to verify.")
