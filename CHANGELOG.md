# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

### Changed

### Deprecated

### Removed

## [0.7.0] - 2025-11-17

### Added

- Real time and finite-temperature evolution functionality for simple update
- Correlator for mixed state `InfinitePEPO`
- `SUWeight` to `CTMRGEnv` conversion (as `InfinitePEPS` environment)
- Simple update for PEPO (including 3-site version)
- Single-layer and double-layer PEPO reduced density matrix
- `spacetype` method for `InfinitePartitionFunction`
- Support for `SU2Irrep` symmetry in `j1_j2_model`

### Fixed

- Add unit normalization of the half and full infinite environments before the projector computation, fixing an issue with the gradient accuracy
- Fix sporadic test timeouts when running variational optimization after simple update for the Heisenberg model by switching to a `GMRES`-based gradient solver
- Rotation of iPEPO is now done correctly
- Fix `rotl90`, `rotr90` and `rot180` for `LocalOperator`
- Fix XXZ model convention
- Fix `add_physical_charge` for fermionic operators
- Fix `maxiter` behavior for fallback gradient linear solver
- Fix gauge fixing in `:fixed` mode for non-uniform unit cells from full SVD

### Changed

- A unified interface for Trotter-based time evolution algorithms. The old `su_iter`, `simpleupdate` functions should be replaced by `timestep`, `time_evolve` respectively
- Default fixed-point gradient algorithm changed to `:eigsolver`
- BoundaryMPS methods now have their own custom transfer functions, avoiding a double conjugation and twist issues for fermions
- `physicalspace` and related functions now correctly handle periodic indexing for infinite networks
- Updated compatibility with TensorKit v0.15
- Restrict Julia to `<v1.12` due to Zygote incompatibility
- Runic formatter

### Removed

- `InfiniteWeightPEPS` and `mirror_antidiag`
- Support for integer space specifiers in state and environment constructors
- Removed redefinition of `tensorexpr`
- Support for dual physical spaces for non-bosonic symmetries

### Performance

- Avoid `@autoopt` for partition function calculations
- Multithreaded scheduler now correctly taken into account for the reverse rules.


[unreleased]: https://github.com/quantumkithub/pepskit.jl/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/quantumkithub/pepskit.jl/compare/v0.6.1...v0.7.0
