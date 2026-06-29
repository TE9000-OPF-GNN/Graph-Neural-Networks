import pickle, os, sys, types

# Stub classes so pickle can deserialise without full torch import
class _Stub:
    def __reduce__(self):
        return (object, ())

# Register stubs for torch.nn.Module subclasses stored in pickle
import importlib
_fake_main = types.ModuleType("__main__")
for cls_name in ["PowerFlowGNN", "PhysicsConfig"]:
    setattr(_fake_main, cls_name, type(cls_name, (), {"__setstate__": lambda self, s: self.__dict__.update(s) if isinstance(s, dict) else None}))
sys.modules["__main__"] = _fake_main

# Also handle torch tensors
try:
    import torch  # noqa: F401
except ImportError:
    pass

p = os.path.join(
    r"C:\Users\STSI\OneDrive - USN\Data_PF_GNN\training_results_saved",
    "sweep_F5_full_curriculum_new.json",
)
with open(p, "rb") as f:
    runs = pickle.load(f)

print(f"Total runs: {len(runs)}\n")

for i, r in enumerate(runs):
    rk = r.get("run_key", "?")
    tag = r.get("tag", "?")
    epochs = r.get("num_epochs", "?")
    h = r.get("history", {})

    frac_keys = sorted([k for k in h if "frac" in k.lower() or "eff_w" in k.lower()])
    flow_keys = sorted([k for k in h if "flow" in k.lower()])

    print(f"--- Run {i}: tag={tag}, epochs={epochs} ---")
    print(f"  run_key: {rk[:120]}")
    print(f"  history keys ({len(h)}): {sorted(h.keys())}")
    print(f"  fraction/eff_w keys: {frac_keys}")
    print(f"  flow keys: {flow_keys}")

    # Effective weight channels
    ew_keys = [
        "train_eff_w_dcf_global",
        "train_eff_w_acf_global",
        "train_eff_w_dcf_local",
        "train_eff_w_acf_local",
        "train_eff_w_physics",
        "train_eff_w_ptdf",
    ]
    for k in ew_keys:
        vals = h.get(k)
        if vals:
            print(f"  {k}: len={len(vals)}, first3={vals[:3]}, last3={vals[-3:]}")
        else:
            print(f"  {k}: MISSING")

    # PhysicsConfig fractions
    pc = r.get("physics_config", {})
    fk_list = [
        "fraction_physics",
        "fraction_ptdf",
        "fraction_dcf_local",
        "fraction_dcf_global",
        "fraction_acf_local",
        "fraction_acf_global",
    ]
    for fk in fk_list:
        v = pc.get(fk, "N/A")
        print(f"  cfg.{fk} = {v}")

    # Activation schedule
    ak_list = [
        "physics_activation",
        "ptdf_activation",
        "dcf_local_activation",
        "dcf_global_activation",
        "acf_local_activation",
        "acf_global_activation",
        "activation_ramp_epochs",
    ]
    for ak in ak_list:
        v = r.get(ak, "N/A")
        print(f"  {ak} = {v}")

    print()
    if i >= 4:
        remaining = len(runs) - 5
        if remaining > 0:
            print(f"... ({remaining} more runs)")
        break
