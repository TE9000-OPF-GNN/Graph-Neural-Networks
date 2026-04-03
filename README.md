# PyPSA Power System Analysis Project

# Power System GNNs on PyPSA

This repo contains research code for training Graph Neural Networks (GNNs) on power system data built with [PyPSA](https://pypsa.org/). The focus is physics‑informed learning (AC residuals, PTDF terms) and robustness to topology changes.

The code is **usable for research**, but it is **not a polished library** yet.

---

## Where things stand

Honest status:

- The end‑to‑end pipeline works: generate networks → build datasets → train GNNs → run hyperparameter sweeps → analyse results.
- A `PhysicsConfig` object drives most physics/PTDF options; some older code still uses earlier patterns and needs cleanup.
- Hyperparameter sweeps support checkpoint/resume and save metrics, histories and timing for later analysis.
- There are smoke tests and a regression‑style baseline test, but test coverage is patchy and not organised as a proper test suite.
- Documentation is minimal; some key behaviour only lives in notebooks.

If you want a stable, versioned Python package right now, this is not that (yet). If you’re happy working with “pretty clean research code” and helping push it towards a package, you’re in the right place.

---

## Medium‑term goals

Rough roadmap:

- **Stabilise experiments**
  - Make `PhysicsConfig` the single source of truth for all physics/PTDF settings.
  - Harden `run_hparam_sweep` (checkpoint/resume, metadata, file formats).
  - Keep a small set of smoke/regression tests as gates for changes.

- **Move towards a package**
  - Switch to a `src/` layout and add `pyproject.toml`.
  - Pull core logic (data generation, datasets, training, eval, sweeps) out of notebooks into importable modules.
  - Define a small public API and back it with tests.

- **Make it runnable in containers / cloud**
  - Add a Dockerfile and simple CLI or script entrypoints.
  - Make it easy to run sweeps and analysis non‑interactively.
  - Document how to run this in environments like Databricks.

More detail and open tasks live in [BACKLOG.md](./BACKLOG.md).

---

## What currently works (in practice)

Today you can:

- Generate topology variants for standard test systems in PyPSA, with controlled perturbations and sanity checks.
- Build PyTorch Geometric graph datasets from those networks (node features and targets tailored to AC power flow).
- Train GNNs with physics‑informed loss terms (AC residuals, optional angle reference and PTDF losses).
- Run hyperparameter sweeps with checkpoint/resume and store per‑run metrics, losses and training time.
- Use helper functions to summarise and plot sweep results, and to run a baseline regression test.

These pieces are designed for experimentation rather than as a stable public interface.

---

## Who this is for (right now)

- People comfortable reading and editing Python + PyTorch + PyPSA code.
- Researchers who want to experiment with GNNs for power systems and don’t mind a bit of plumbing.
- Contributors who’d like to help turn this into:
  - a usable Python package on top of PyPSA (and maybe PowSybl),
  - a reproducible experiment framework,
  - something that can run cleanly inside a container.

---

## Getting started

Very short version:

1. Clone the repo and set up a Python environment.
2. Run a small data‑generation script/notebook to produce a toy dataset.
3. Run a short training or sweep (few configs) to check everything works.
4. Use the analysis helpers to inspect results.

The exact entry scripts/notebooks may change as the repo is cleaned up; check the issues and backlog for current pointers.

---

## Contributing

Contributions are welcome, especially around:

- pulling core logic into a package structure,
- improving tests and adding CI,
- clarifying docs and examples,
- containerisation and cloud deployment patterns.

Before starting larger work, please open an issue to discuss scope and direction. Small, focused PRs are easier to review and merge.

---

## License

to be decided
