# Project backlog and known issues

This document is the working backlog for the current research/dev codebase.

Its purpose is to make collaboration easier by separating:
- implemented but not fully hardened work,
- known issues and technical debt,
- postponed tasks,
- packaging/containerization roadmap items.

This project is currently research-first code, but the medium-term direction is:
1. a cleaner Python package on top of PyPSA, with possible PowSybl support later,
2. reproducible experiment execution,
3. containerized execution for cloud environments such as Databricks.

---

## Status labels

Use these labels in issues and PRs:
- `bug`: incorrect behavior or broken functionality
- `tech-debt`: code works but needs cleanup/refactor
- `tests`: missing or weak test coverage
- `docs`: missing or unclear documentation
- `packaging`: Python package structure and installability
- `container`: Docker / runtime environment / Databricks
- `research`: experiment-only or exploratory work
- `postponed`: explicitly deferred, not forgotten
- `good-first-issue`: safe entry point for collaborators
- `help-wanted`: good target for external contribution

---

## What is already implemented

- Generation of trainingsets with PyPSA based on a suite of IEEE test networks (IEEE 9, 30, 39, 57 and 118 bus)
- Topology variations across generator/load buses, number of buses, line parameters, gen-setpoints, random walk load variations (congfigurable time steps)
- Generation of training datasets based on these generated network for training of both pure GNN and physics informed GNN
- Parameter sweep for comparing different variants of parameterization
- factorial analysis and visual comparison of metrics
- Physics/PTDF configuration has been moved toward a config-driven approach centered on `PhysicsConfig`.
- Hyperparameter sweep code supports checkpoint/resume behavior.
- Sweep runs persist timing and metadata useful for later analysis.
- PTDF-related dataset/training support has been extended beyond the earlier simpler setup.
- Result saving/loading and metrics export utilities exist, but still need cleanup and standardization.

---

## Immediate priorities

### 1. Stabilize experiment pipeline
- [ ] Verify that `run_hparam_sweep` uses `PhysicsConfig` as the single source of truth for physics/PTDF settings everywhere.
- [ ] Remove any remaining older code paths that still pass PTDF settings separately.
- [ ] Ensure checkpoint/resume works reliably after interruption or partial failure.
- [ ] Standardize saved run metadata fields and naming.

### 2. Strengthen test coverage
- [ ] Add a tiny checkpoint/resume smoke test.
- [ ] Add a legacy checkpoint backfill test for older saved runs missing `run_key`.
- [ ] Add a checkpoint schema/content test for required fields.
- [ ] Keep and formalize a baseline regression test for accepted reference metrics.
- [ ] Keep and formalize all-systems data-generation smoke tests.

### 3. Clean separation of dev vs run workflows
- [ ] Remove implementation-debug cells/prints from the clean experiment notebook.
- [ ] Keep only sweep execution and result-analysis logic in the run notebook.
- [ ] Move deep debugging helpers to a dev notebook or test module.
- [ ] Remove or archive obsolete helper functions such as old sweep interfaces.

### 4. Training improvements
- [ ] Constant learning rate (no decay)
- [ ] Early stopp if training and validation converges (All runs to max epochs now)
- [ ] GPU and/or parallellization of training to improve training time (neccessary for larger networks...)
---

## Known issues

### Configuration consistency
- [ ] Some analysis/export helpers may still assume older sweep field structures.
- [ ] Some plotting/reporting helpers likely need refactoring to align with the new config model.

### Checkpointing and persistence
- [ ] Full run pickles may be too heavy because they can include models and histories.
- [ ] Need a decision on whether checkpoints should store:
  - full Python objects,
  - metrics only,
  - or a split format (light checkpoint + optional model export).


### Code organization
- [ ] Research code, utility code, notebook code, and packaging candidates are still mixed together.
- [ ] Core logic is not yet cleanly separated from experiment orchestration.
- [ ] There is not yet a stable public API for external users.

### Documentation
- [ ] Missing collaborator-facing architecture overview.
- [ ] Missing “how to run a sweep” guide.
- [ ] Missing “results format” documentation.
- [ ] Missing contributor guidance on which notebook/script is authoritative.

---

## Explicitly postponed

These items are intentionally postponed, not forgotten.

### Short-term postponed
- [ ] Full cleanup of all legacy helper names and backward-compatibility paths.
- [ ] Refactoring all notebook utilities into package modules.
- [ ] Reworking all plotting/analysis helpers for a polished public API.
- [ ] Converting all regression utilities into formal pytest suites.

### Medium-term postponed
- [ ] Split package design for:
  - data/topology generation,
  - dataset building,
  - training,
  - evaluation,
  - sweep orchestration,
  - results analysis.
- [ ] Optional backend abstraction for PyPSA and PowSybl.
- [ ] CLI entrypoints for experiments and evaluation.
- [ ] Robust configuration management (YAML/TOML/CLI-driven runs).

---

## Packaging roadmap

Target: turn the codebase into an installable Python package suitable for reuse and testing.

### Package structure
- [ ] Move to a `src/` layout.
- [ ] Add/update `pyproject.toml`.
- [ ] Define package boundaries and public modules.
- [ ] Create a minimal install path with optional extras:
  - `core`
  - `dev`
  - `pypsa`
  - `powsybl`
  - `viz`
- [ ] Reduce notebook-only logic inside package modules.
- [ ] Add versioning strategy.

### Public API design
- [ ] Identify stable entry points:
  - topology/data generation,
  - dataset creation,
  - model training,
  - evaluation,
  - sweep execution.
- [ ] Decide what remains internal/private.
- [ ] Add typed interfaces and clearer dataclasses/configs.

### Testability
- [ ] Make core functions pure or close to pure where possible.
- [ ] Reduce hidden global state and side effects.
- [ ] Separate file I/O from compute logic.
- [ ] Add install-based tests that import the package rather than relying on notebook state.

---

## Containerization roadmap

Target: reproducible execution in Docker and cloud runtimes, including Databricks-oriented workflows.

### Container basics (Suggestion, to be decided)
- [ ] Add a production-oriented Dockerfile.
- [ ] Use a small Python base image.
- [ ] Consider a multi-stage build.
- [ ] Pin runtime dependencies.
- [ ] Run as a non-root user where practical.
- [ ] Make logs/stdout behavior container-friendly.

### Databricks readiness (Suggestiong, to be decided)
- [ ] Define one reproducible batch entrypoint for training/sweeps.
- [ ] Define one entrypoint for post-run analysis/export.
- [ ] Ensure outputs can be written to mounted/cloud paths cleanly.
- [ ] Avoid notebook-only assumptions in execution logic.
- [ ] Document environment variables and expected filesystem layout.



### Reproducibility
- [ ] Make random seed handling explicit and centralized.
- [ ] Record environment/package versions in outputs.
- [ ] Store enough metadata to reproduce a run from saved results.

## Future research goals
Use GNN inference/predictions as surrogate for PF in contingency simulations
### Extension for stochastic simulation (Neccessary for further research - might be a new package related to this)
- [ ] Implement contingency logic in training (Guided Dropout or similar?)
- [ ] Implement open-loop version contingency logic must be expanded with best-backup choice (optimization method to be decided)
- [ ] Implement sampling from failure probability models for stochastic contingency simulariong
- [ ] Implement probabilistic load per node based on model (model and parameters to be decided)
- [ ] Implement dynamic line rating model for lines, cables and transformers.
- [ ] Implement stochastic simulation of all three variables and resulting limit violations

---

## Collaboration tasks

### Repo hygiene
- [ ] Add `CONTRIBUTING.md`.
- [ ] Add issue templates for bug reports, feature requests, and research tasks.
- [ ] Add PR template.
- [ ] Add `CODE_OF_CONDUCT.md` if the repo will be open to wider external collaboration.
- [ ] Add a clear roadmap section in `README.md`.

### Recommended contributor entry points
- [ ] Testing and smoke-test formalization.
- [ ] Packaging refactor with `src/` layout.
- [ ] Results schema cleanup.
- [ ] Documentation and examples.
- [ ] Container/Docker setup.

---

## Suggested near-term milestone

### Milestone: Collaboration-ready research repo
- [ ] Clean run notebook
- [ ] Working checkpoint/resume sweep
- [ ] Stable saved run schema
- [ ] Smoke tests + baseline regression test
- [ ] README with project status and roadmap
- [ ] CONTRIBUTING + issue templates

### Milestone: Package-ready core
- [ ] `src/` layout
- [ ] `pyproject.toml`
- [ ] core modules extracted from notebooks/scripts
- [ ] pytest-based test suite
- [ ] documented public API

### Milestone: Container-ready execution
- [ ] Dockerfile
- [ ] CLI or script entrypoint
- [ ] reproducible config-driven run mode
- [ ] documented output locations
- [ ] Databricks execution notes

---

## Notes for collaborators

This repository currently contains research code under active cleanup. Some parts are already functional and useful, but not all components are yet packaged, stabilized, or uniformly documented.

If you want to contribute:
- prefer small, well-scoped PRs,
- preserve backward compatibility for saved experiment results where practical,
- avoid introducing new parallel config paths,
- prioritize testability and reproducibility over convenience hacks.