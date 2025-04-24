# Models

PEPSKit implements physical models through the [MPSKitModels.jl](https://quantumkithub.github.io/MPSKitModels.jl/dev/) package as [`LocalOperator`](@ref) structs.
Here, we want to explain how users can define their own Hamiltonians and provide a list of
already implemented models.

## Implementing custom models

In order to define custom Hamiltonians, we leverage several of the useful tools provided in MPSKitModels.
In particular, we use many of the pre-defined [operators](@extref MPSKitModels Operators), which is especially useful when defining models with symmetric and fermionic tensors, since most of these operators can take a symmetry as an argument, returning the appropriate symmetric `TensorMap`.
In order to specify the lattice on which the Hamiltonian is defined, we construct two-dimensional lattices as subtypes of [`MPSKitModels.AbstractLattice`](@extref).
Note that so far, all models are defined on infinite square lattices, see [`InfiniteSquare`](@ref), but in the future, we plan to support other lattice geometries as well.
In order to specify tensors acting on particular lattice sites, there are a couple of handy methods that we want to point to: see `vertices`, `nearest_neighbors` and `next_nearest_neighbors` defined [here](https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/src/operators/lattices/squarelattice.jl).

For a simple example on how to implement a custom model, let's look at the implementation of the [`MPSKitModels.transverse_field_ising`](@extref) model:

```julia
function MPSKitModels.transverse_field_ising(
    T::Type{<:Number},
    S::Union{Type{Trivial},Type{Z2Irrep}},
    lattice::InfiniteSquare;
    J=1.0,
    g=1.0,
)
    ZZ = rmul!(σᶻᶻ(T, S), -J)
    X = rmul!(σˣ(T, S), g * -J)
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => ZZ for neighbor in nearest_neighbours(lattice))...,
        ((idx,) => X for idx in vertices(lattice))...,
    )
end
```

This provides a good recipe for defining a model:

1. Define the locally-acting tensors as `TensorMap`s.
2. Construct a matrix of the physical spaces these `TensorMap`s act on based on the lattice geometry.
3. Return a `LocalOperator` where we specify on which sites (e.g. on-site, nearest neighbor, etc.) the local tensors act.

For more model implementations, check the [PEPSKit repository](https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/src/operators/models.jl).

## Implemented models

While PEPSKit provides an interface for specifying custom Hamiltonians, it also provides a number of [pre-defined models](https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/src/operators/models.jl). Some of these are models already defined in [MPSKitModels](@extref MPSKitModels Models), which are overloaded for two-dimensional lattices and re-exported, but there are new additions as well. The following models are provided:

### MPSKitModels.jl models

```@docs
MPSKitModels.transverse_field_ising
MPSKitModels.heisenberg_XYZ
MPSKitModels.heisenberg_XXZ
MPSKitModels.hubbard_model
MPSKitModels.bose_hubbard_model
MPSKitModels.tj_model
```

### PEPSKit.jl models

```@docs
j1_j2_model
pwave_superconductor
```
