# Models

PEPSKit implements physical models as [`PEPSKit.LocalOperator`](@ref) structs.
Here, we want to explain how users can define their own Hamiltonians and provide a list of
already implemented models.

## Implementing custom models

In order to define custom Hamiltonians, we leverage the operator building blocks provided by
[TensorKitTensors.jl](https://quantumkithub.github.io/TensorKitTensors.jl/stable/), which offers
pre-defined symmetric tensors for spin, boson, fermion, and Hubbard systems.
In order to specify the lattice on which the Hamiltonian is defined, we construct two-dimensional lattices as subtypes of [`MPSKitModels.AbstractLattice`](@extref).
Note that so far, all models are defined on infinite square lattices, see [`InfiniteSquare`](@ref), but in the future, we plan to support other lattice geometries as well.
In order to specify tensors acting on particular lattice sites, there are a couple of handy methods that we want to point to: see `vertices`, `nearest_neighbors` and `next_nearest_neighbors` defined [here](https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/src/operators/lattices/squarelattice.jl).

For a simple example on how to implement a custom model, let's look at the implementation of the
[`transverse_field_ising`](@ref) model:

```julia
import TensorKitTensors.SpinOperators as SO
function transverse_field_ising(
    T::Type{<:Number},
    S::Union{Type{Trivial},Type{Z2Irrep}},
    lattice::InfiniteSquare;
    J=1.0,
    g=1.0,
)
    ZZ = rmul!(SO.S_z_S_z(T, S), -4 * J)
    X = rmul!(SO.σˣ(T, S), g * -J)
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => ZZ for neighbor in nearest_neighbours(lattice))...,
        ([idx,] => X for idx in vertices(lattice))...,
    )
end
```

This provides a good recipe for defining a model:

1. Define the locally-acting tensors as `TensorMap`s.
2. Construct a matrix of the physical spaces these `TensorMap`s act on based on the lattice geometry.
3. Return a `LocalOperator` where we specify on which sites (e.g. on-site, nearest neighbor, etc.) the local tensors act.

For more model implementations, check the [PEPSKit repository](https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/src/operators/models.jl).

## Implemented models

PEPSKit provides a number of pre-defined models. The following model constructors are available
and can be used directly with an [`InfiniteSquare`](@ref) lattice:

### Models inherited from MPSKitModels.jl

```@docs; canonical=false
transverse_field_ising(::InfiniteSquare)
heisenberg_XYZ(::InfiniteSquare)
heisenberg_XXZ(::InfiniteSquare)
hubbard_model(::InfiniteSquare)
bose_hubbard_model(::InfiniteSquare)
tj_model(::InfiniteSquare)
```

### Models introduced by PEPSKit.jl

```@docs
j1_j2_model
pwave_superconductor
```
