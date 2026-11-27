"""
$(TYPEDEF)

Infinite triangular lattice with a unit cell of size `(Nrows, Ncols)`, optionally combined
with a sublattice rotation.

The lattice is embedded in a matrix of sites `[r, c]`, where each site connects to the six
neighbors at offsets `(-1, 0)`, `(-1, 1)`, `(0, 1)`, `(1, 0)`, `(1, -1)` and `(0, -1)`,
corresponding to the `NW`, `NE`, `E`, `SE`, `SW` and `W` virtual legs of an
[`InfinitePEPSTriangular`](@ref). Moving along a row corresponds to the lattice vector at 0°,
and moving down a column to the lattice vector at -60°.

## Sublattice rotation

A nonzero `rotation = (θr, θc)` describes the spiral ansatz of
[Hasik and Corboz](https://arxiv.org/abs/2311.05534), in which the physical state is
```math
|ψ⟩ = U |ψ_{\\mathrm{PEPS}}⟩, \\qquad U = \\prod_{[r, c]} \\exp\\left(i φ_{r, c} S^α_{r, c}\\right),
\\qquad φ_{r, c} = θ_r (r - 1) + θ_c (c - 1),
```
where ``S^α`` is the spin operator along `axis` (`:x`, `:y` or `:z`). Model constructors
such as [`heisenberg_XYZ`](@ref) then return the rotated operator ``U^† H U``, which is
expressed on the `(Nrows, Ncols)` unit cell of ``|ψ_{\\mathrm{PEPS}}⟩``. For terms that are
invariant under a global rotation about `axis` (e.g. any SU(2)-symmetric interaction), the
rotated term only depends on the relative angles, such that `(Nrows, Ncols) = (1, 1)` can be
used for any (possibly incommensurate) rotation. Other terms require the rotation to be
commensurate with the `(Nrows, Ncols)` unit cell. See [`rotate_sublattices`](@ref) for
rotating arbitrary operators and for converting the PEPS to an explicit representation of
``|ψ⟩``.

Some common choices for a single-site unit cell are
- `rotation = (-2π/3, 2π/3)`: 120° order on the three sublattices `[A B C; C A B; B C A]`,
  i.e. the state of [`InfiniteTriangularThreeSublattice`](@ref) with `B` and `C` the
  rotated versions of `A`;
- `rotation = (π, π)`: the two-sublattice structure `[A B; B A]`.

## Fields

$(TYPEDFIELDS)

## Constructor

    InfiniteTriangular([Nrows=1, Ncols=1]; rotation=(0, 0), axis=:y)

By default, an infinite triangular lattice with a (1, 1)-unitcell and no sublattice rotation
is constructed.
"""
struct InfiniteTriangular <: AbstractLattice{2}
    Nrows::Int
    Ncols::Int
    "rotation angles per unit step along the rows (`r`) and columns (`c`)"
    rotation::NTuple{2, Float64}
    "spin axis generating the sublattice rotation"
    axis::Symbol
    function InfiniteTriangular(
            Nrows::Integer = 1, Ncols::Integer = 1;
            rotation::Tuple{Real, Real} = (0, 0), axis::Symbol = :y,
        )
        Nrows > 0 && Ncols > 0 || error("unit cell size needs to be positive")
        axis in (:x, :y, :z) || throw(ArgumentError("rotation axis should be `:x`, `:y` or `:z`"))
        return new(Nrows, Ncols, Float64.(rotation), axis)
    end
end

"""
    has_sublattice_rotation(lattice) -> Bool

Whether `lattice` carries a nontrivial sublattice rotation.
"""
has_sublattice_rotation(lattice) = false
has_sublattice_rotation(lattice::InfiniteTriangular) = any(!iszero, lattice.rotation)

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

If the three sublattices are related by a spin rotation (as for 120° order in SU(2)-symmetric
models), a single tensor suffices: see the `rotation` of [`InfiniteTriangular`](@ref), e.g.
`InfiniteTriangular(1, 1; rotation = (-2π/3, 2π/3))`.

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
