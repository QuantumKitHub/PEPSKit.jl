"""
$(TYPEDEF)

Infinite triangular lattice with a unit cell of size `(Nrows, Ncols)`.

The lattice is embedded in a matrix of sites `[r, c]`, where each site connects to the six
neighbors at offsets `(-1, 0)`, `(-1, 1)`, `(0, 1)`, `(1, 0)`, `(1, -1)` and `(0, -1)`,
corresponding to the `NW`, `NE`, `E`, `SE`, `SW` and `W` virtual legs of an
[`InfinitePEPSTriangular`](@ref). Moving along a row corresponds to the lattice vector at 0°,
and moving down a column to the lattice vector at -60°.

## Fields

$(TYPEDFIELDS)

## Constructor

    InfiniteTriangular([Nrows=1, Ncols=1])

By default, an infinite triangular lattice with a (1, 1)-unitcell is constructed.
"""
struct InfiniteTriangular <: AbstractLattice{2}
    Nrows::Int
    Ncols::Int
    function InfiniteTriangular(Nrows::Integer = 1, Ncols::Integer = 1)
        Nrows > 0 && Ncols > 0 || error("unit cell size needs to be positive")
        return new(Nrows, Ncols)
    end
end

Base.size(lattice::InfiniteTriangular) = (lattice.Nrows, lattice.Ncols)

function vertices(lattice::InfiniteTriangular)
    return CartesianIndices((1:(lattice.Nrows), 1:(lattice.Ncols)))
end

"""
$(TYPEDEF)

Infinite triangular lattice tiled by a three-site unit cell, where the three sublattices
`A`, `B` and `C` form an elementary triangle and every triangle of the lattice contains one
site of each sublattice (the √3 × √3 structure).

In the `[r, c]` coordinates of [`InfiniteTriangular`](@ref), site `[r, c]` belongs to
sublattice `mod(c - r, 3) + 1` (`1 => A`, `2 => B`, `3 => C`), giving the pattern
`[A B C; C A B; B C A]`. Operators are defined on this `(3, 3)` cell, but only contain the
bonds starting from the representative sites `(1, 1)`, `(1, 2)` and `(1, 3)` of `A`, `B`
and `C`, such that they measure quantities per three-site unit cell.

A matching state stores only the three tensors `A`, `B` and `C`, in the sheared `(1, 3)`
unit cell `[A B C]` with `shift = 1` (see [`TriangularUnitCell`](@ref)), e.g.
```julia
peps = InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace; unitcell = (1, 3), shift = 1)
env = CTMRGEnvTriangular(randn, ComplexF64, Vspace ⊗ Vspace', χspace; unitcell = (1, 3), shift = 1)
```
Optimizing such a state with [`fixedpoint`](@ref) directly yields the gradient with respect
to the three independent tensors.

## Constructor

    InfiniteTriangularThreeSublattice()
"""
struct InfiniteTriangularThreeSublattice <: AbstractLattice{2} end

Base.size(::InfiniteTriangularThreeSublattice) = (3, 3)

vertices(::InfiniteTriangularThreeSublattice) = CartesianIndex.([(1, 1), (1, 2), (1, 3)])

# bonds in the 0°, 60° and 120° directions, matching the triangular `energy` contractions
function nearest_neighbours(lattice::Union{InfiniteTriangular, InfiniteTriangularThreeSublattice})
    neighbors = Vector{CartesianIndex{2}}[]
    for idx in vertices(lattice)
        push!(neighbors, [idx, idx + CartesianIndex(0, 1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, -1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, 0)])
    end
    return neighbors
end

# bonds of length √3 in the 330°, 270° and 210° directions, connecting equal sublattices
function next_nearest_neighbours(lattice::Union{InfiniteTriangular, InfiniteTriangularThreeSublattice})
    neighbors = Vector{CartesianIndex{2}}[]
    for idx in vertices(lattice)
        push!(neighbors, [idx, idx + CartesianIndex(1, 1)])
        push!(neighbors, [idx, idx + CartesianIndex(2, -1)])
        push!(neighbors, [idx, idx + CartesianIndex(1, -2)])
    end
    return neighbors
end
