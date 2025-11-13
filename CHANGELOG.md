# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

- Add unit normalization of the half and full infinite environments before the projector
  computation, fixing an issue with the gradient accuracy.
- Fix sporadic test timeouts when running variational optimization after simple
  update for the Heisenberg model by switching to a `GMRES`-based gradient solver.

### Changed

### Deprecated

### Removed


[unreleased]: https://github.com/quantumkithub/pepskit.jl/compare/v0.6.1...HEAD