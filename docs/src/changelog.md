# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Guidelines for updating this changelog

When making changes to this project, please update the "Unreleased" section with your changes under the appropriate category:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Performance** for performance improvements.

When releasing a new version, move the "Unreleased" changes to a new version section with the release date.

## [Unreleased](https://github.com/quantumkithub/pepskit.jl/compare/v0.8.0...HEAD)

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Performance

## [0.8.0](https://github.com/quantumkithub/pepskit.jl/compare/v0.7.0...v0.8.0) - 2026-05-08

### Added

- C4v CTMRG support, including QR-CTMRG variant (#321, #329)
- Belief propagation gauge fixing, with bipartite specialization and support for purified iPEPO (#223, #318, #319)
- N-site Simple Update (#339)
- `LocalCircuit` operator type, alongside refactor of `LocalOperator` operator types (#347)
- Rotation of `LocalCircuit` (#349)

### Changed

- Bump OptimKit.jl compatibility to v0.4
- Move `info.truncation_error` and `info.condition_number` into the `info.contraction_metrics` named tuple for `leading_boundary` and `fixedpoint`
- Return `info.converged` flag and `info.convergence_error` in `leading_boundary` named tuple
- Rename algorithm symbols (#376)
- Rename `bondenv_fu` to `bondenv_ctm` (#343)
- `fixedpoint` improvements; relax `env₀` type restriction in `select_algorithm` (#337, #345)
- Refactor simple update / reorganize trotter and apply_gate code (#338, #346). Each term of the input Hamiltonian `LocalOperator` is now exponentiated individually when constructing the Trotter gates in the resulting `LocalCircuit`.
- Update decomposition handling (#364)
- Use MatrixAlgebraKit SVD pullbacks (#335)
- Bump TensorKit compat to 0.16 (#314)
- Make `SUWeight` axis order definite (#315)
- Improve Vidal gauge conversion and CTM bond env for PEPO (#348)
- Improve bond truncation algorithms (#303). In particular, the 3-leg reduced bond tensors now follows the leg order convention of an MPS.
- Periodic indexing improvements (#377)

### Removed

- `:diffgauge` differentiation mode, also for eigh-based C4v CTMRG (#334, #370)
- Unused implementations of `LinearAlgebra` methods for `CTRMGEnv` and `InfinitePEPS`

### Fixed

- Ensure MPSKit overloads accept all keywords to intercept all dispatches (#374)
- `FixedSpaceTruncation` for simple update (#360)
- Stack overflow in `renormalize_southwest_corner` (#344)
- Broken gradients for C4v eigh-CTMRG (#333)
- CTMRG contraction inconsistencies (#327)
- Typo in `edge_transfer_right` (#310)
- MPS cluster truncation with standard virtual arrows (#309)
- Small fixes for `InfiniteSquareNetwork` and `InfinitePartitionFunction` (#306)
- Zero gate returned by `get_gateterm` (#300)

### Performance

- Improve simple update performance (#361)
- Improve efficiency of bond truncation (#366)

## [0.7.0](https://github.com/quantumkithub/pepskit.jl/compare/v0.6.1...v0.7.0) - 2025-11-17

### Added

- Real time and finite-temperature evolution functionality for simple update
- Correlator for mixed state `InfinitePEPO`
- `SUWeight` to `CTMRGEnv` conversion (as `InfinitePEPS` environment)
- Simple update for PEPO (including 3-site version)
- Single-layer and double-layer PEPO reduced density matrix
- `spacetype` method for `InfinitePartitionFunction`
- Support for `SU2Irrep` symmetry in `j1_j2_model`

### Changed

- A unified interface for Trotter-based time evolution algorithms. The old `su_iter`, `simpleupdate` functions should be replaced by `timestep`, `time_evolve` respectively
- Default fixed-point gradient algorithm changed to `:EigSolver`
- BoundaryMPS methods now have their own custom transfer functions, avoiding a double conjugation and twist issues for fermions
- `physicalspace` and related functions now correctly handle periodic indexing for infinite networks
- Updated compatibility with TensorKit v0.15
- Runic formatter

### Removed

- `InfiniteWeightPEPS` and `mirror_antidiag`
- Support for integer space specifiers in state and environment constructors
- Removed redefinition of `tensorexpr`
- Support for dual physical spaces for non-bosonic symmetries

### Fixed

- Add unit normalization of the half and full infinite environments before the projector computation, fixing an issue with the gradient accuracy
- Fix sporadic test timeouts when running variational optimization after simple update for the Heisenberg model by switching to a `GMRES`-based gradient solver
- Rotation of iPEPO is now done correctly
- Fix `rotl90`, `rotr90` and `rot180` for `LocalOperator`
- Fix XXZ model convention
- Fix `add_physical_charge` for fermionic operators
- Fix `maxiter` behavior for fallback gradient linear solver
- Fix gauge fixing in `:fixed` mode for non-uniform unit cells from full SVD


### Performance

- Avoid `@autoopt` for partition function calculations
- Multithreaded scheduler now correctly taken into account for the reverse rules.
