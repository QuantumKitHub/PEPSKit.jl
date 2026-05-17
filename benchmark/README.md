# PEPSKit benchmarks

A [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)-based suite that exercises
performance-critical paths in PEPSKit. The suite is organised into modules; the only module
currently registered is `ctmrg`, which times a single `ctmrg_iteration` across several CTMRG
algorithms, problem scenarios, and symmetry sectors.

## Layout

```
bench/benchmark/
├── benchmarks.jl                 # entry point: builds the top-level SUITE
├── Project.toml                  # benchmark environment (depends on local PEPSKit)
└── PEPSKitBenchmarks/
    ├── PEPSKitBenchmarks.jl      # module registry (MODULES), load!, loadall!
    ├── utils/BenchUtils.jl       # TOML <-> VectorSpace conversion helpers
    └── ctmrg/
        ├── CTMRGBenchmarks.jl    # populates SUITE["ctmrg"]
        ├── ctmrg_iteration_benchmarks.jl
        ├── default.toml          # Trivial + NonUniform scenarios
        └── su3_hubbard.toml      # SU(3) fermionic scenarios
```

New modules are registered in the `MODULES` dict in `PEPSKitBenchmarks.jl`.

## Quickstart

Run the full suite from the repository root:

```sh
julia --project=bench/benchmark -e '
  using Pkg; Pkg.instantiate()
  include("bench/benchmark/benchmarks.jl")
  results = run(SUITE; verbose=true)
'
```

Default per-benchmark budget is 60 s, 5 samples, 1 evaluation (set in `PEPSKitBenchmarks.jl`).
The full suite — in particular the `su3_hubbard` scenarios — is heavy; expect it to run for a
long time. For development, prefer the selective runs below.

## Selective runs

### By module (CLI flag)

`benchmarks.jl` accepts `--modules=<id>[,<id>…]` to load only specific groups:

```sh
julia --project=bench/benchmark bench/benchmark/benchmarks.jl --modules=ctmrg
```

The script only populates `SUITE`; call `run(SUITE)` afterwards (e.g. from an interactive
session) to execute. Available IDs are the keys of `MODULES` in `PEPSKitBenchmarks.jl`.

### By group key

`SUITE` is a nested `BenchmarkGroup` indexed as:

```
module → scenario → algorithm → symmetry → benchname
```

- `scenario` is the TOML basename (`default`, `su3_hubbard`).
- `algorithm` is one of `SequentialCTMRG_HalfInfinite`, `SimultaneousCTMRG_HalfInfinite`.
- `symmetry` is the top-level table name in the TOML (`Trivial`, `NonUniform`, …).
- `benchname` is `"<rows>x<cols>_D<Dmin>_chi<chimin>"`, where `Dmin` is the minimum virtual
  bond dimension and `chimin` is the minimum environment dimension across the unit cell (see
  `benchname` in `ctmrg/ctmrg_iteration_benchmarks.jl`).

To run a single case interactively:

```julia
include("bench/benchmark/benchmarks.jl")
run(SUITE["ctmrg"]["default"]["SimultaneousCTMRG_HalfInfinite"]["Trivial"]["1x1_D2_chi8"])
```

## Extending the suite

### Add a scenario

Scenarios are TOML files in `PEPSKitBenchmarks/ctmrg/`, parsed into `CTMRGSpec` values. Each
top-level table name becomes a symmetry group. Every spec must set exactly these keys:
`unitcell`, `Pspaces`, `Nspaces`, `Espaces`, `chi_north`, `chi_east`, `chi_south`,
`chi_west`. A minimal uniform spec is:

```toml
[[Trivial]]
unitcell = [2, 2]
Pspaces = "ℂ^2"
Nspaces = "ℂ^2"
Espaces = "ℂ^2"
chi_north = "ℂ^8"
chi_east = "ℂ^8"
chi_south = "ℂ^8"
chi_west = "ℂ^8"
```

Each value may be a scalar string (broadcast across the unit cell) or a `[rows][cols]`
array of space strings (per-site). See the `NonUniform` entries in `default.toml` for
worked examples mixing the two forms.

To add a brand-new scenario file, drop it into `ctmrg/` and add it to the `allparams` `Dict`
in `CTMRGBenchmarks.jl`:

```julia
const allparams = Dict(
    "default"     => TOML.parsefile(joinpath(@__DIR__, "default.toml")),
    "su3_hubbard" => TOML.parsefile(joinpath(@__DIR__, "su3_hubbard.toml")),
    "my_scenario" => TOML.parsefile(joinpath(@__DIR__, "my_scenario.toml")),
)
```

### Add a benchmark module

1. Create `PEPSKitBenchmarks/<name>/<Name>Benchmarks.jl` defining a module with
   `const SUITE = BenchmarkGroup()` and populate it.
2. `include(...)` the file from `PEPSKitBenchmarks.jl`.
3. Register the module in `MODULES`:

   ```julia
   const MODULES = Dict{String, Symbol}(
       "ctmrg" => :CTMRGBenchmarks,
       "<name>" => :<Name>Benchmarks,
   )
   ```

The new module is then loadable via `--modules=<name>` and addressable as `SUITE["<name>"]`.
