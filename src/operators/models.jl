#
## Tools for defining Hamiltonians
#

"""
    nearest_neighbour_hamiltonian(lattice::Matrix{S}, h::AbstractTensorMap{T,S,2,2}) where {S,T}

Create a nearest neighbor `LocalOperator` by specifying the 2-site interaction term `h`
which acts both in horizontal and vertical direction.
"""
function nearest_neighbour_hamiltonian(
        lattice::Matrix{S}, h::AbstractTensorMap{T, S, 2, 2}
    ) where {S, T}
    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        J1 = I + CartesianIndex(1, 0)
        J2 = I + CartesianIndex(0, 1)
        push!(terms, [I, J1] => h)
        push!(terms, [I, J2] => h)
    end
    return LocalOperator(lattice, terms...)
end

#
## Model definitions
#

function transverse_field_ising(
        T::Type{<:Number}, S::Union{Type{Trivial}, Type{Z2Irrep}}, lattice::InfiniteSquare;
        J = 1.0, g = 1.0,
    )
    ZZ = rmul!(4 * SO.S_z_S_z(T, S), -J)
    X = rmul!(SO.σˣ(T, S), g * -J)
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => ZZ for neighbor in nearest_neighbours(lattice))...,
        ([idx] => X for idx in vertices(lattice))...,
    )
end

function heisenberg_XYZ(lattice::InfiniteSquare; kwargs...)
    return heisenberg_XYZ(ComplexF64, Trivial, lattice; kwargs...)
end
function heisenberg_XYZ(
        T::Type{<:Number}, S::Type{<:Sector}, lattice::InfiniteSquare;
        Jx = -1.0, Jy = 1.0, Jz = -1.0, spin = 1 // 2,
    )
    term =
        rmul!(SO.S_x_S_x(T, S; spin = spin), Jx) +
        rmul!(SO.S_y_S_y(T, S; spin = spin), Jy) +
        rmul!(SO.S_z_S_z(T, S; spin = spin), Jz)
    spaces = fill(domain(term)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces, (neighbor => term for neighbor in nearest_neighbours(lattice))...
    )
end

function heisenberg_XXZ(
        T::Type{<:Number}, S::Type{<:Sector}, lattice::InfiniteSquare;
        J = 1.0, Delta = 1.0, spin = 1
    )
    h =
        J * (
        (SO.S_plus_S_min(T, S; spin = spin) + SO.S_min_S_plus(T, S; spin = spin)) / 2 +
            Delta * SO.S_z_S_z(T, S; spin = spin)
    )
    spaces = fill(domain(h)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces, (neighbor => h for neighbor in nearest_neighbours(lattice))...
    )
end

function hubbard_model(
        T::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector},
        lattice::InfiniteSquare;
        t = 1.0, U = 1.0, mu = 0.0, n::Integer = 0,
    )
    # TODO: just add this
    @assert n == 0 "Currently no support for imposing a fixed particle number"
    N = Hub.e_num(T, particle_symmetry, spin_symmetry)
    pspace = space(N, 1)
    unit = TensorKit.id(pspace)
    hopping = Hub.e_hopping(T, particle_symmetry, spin_symmetry)
    interaction_term = Hub.ud_num(T, particle_symmetry, spin_symmetry)
    site_term = U * interaction_term - mu * N
    h = (-t) * hopping + (1 / 4) * (site_term ⊗ unit + unit ⊗ site_term)
    return nearest_neighbour_hamiltonian(fill(pspace, size(lattice)), h)
end

function bose_hubbard_model(
        T::Type{<:Number}, symmetry::Type{<:Sector}, lattice::InfiniteSquare;
        cutoff::Integer = 5, t = 1.0, U = 1.0, mu = 0.0, n::Integer = 0,
    )
    hopping_term = BO.b_hopping(T, symmetry; cutoff)
    N = BO.b_num(T, symmetry; cutoff = cutoff)
    interaction_term = N * (N - id(domain(N)))

    spaces = fill(space(N, 1), (lattice.Nrows, lattice.Ncols))

    H = LocalOperator(
        spaces,
        (neighbor => -t * hopping_term for neighbor in nearest_neighbours(lattice))...,
        ([idx] => U / 2 * interaction_term - mu * N for idx in vertices(lattice))...,
    )

    if symmetry === Trivial
        iszero(n) || throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif symmetry === U1Irrep
        isinteger(2n) ||
            throw(ArgumentError("`U₁` symmetry requires halfinteger particle number"))
        H = MPSKit.add_physical_charge(H, fill(U1Irrep(n), size(spaces)...))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end

function tj_model(
        T::Type{<:Number}, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector},
        lattice::InfiniteSquare;
        t = 2.5, J = 1.0, mu = 0.0, slave_fermion::Bool = false,
    )
    hopping = tJ.e_hopping(T, particle_symmetry, spin_symmetry; slave_fermion)
    num = tJ.e_number(T, particle_symmetry, spin_symmetry; slave_fermion)
    heis = tJ.S_exchange(T, particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
    pspace = space(num, 1)
    unit = TensorKit.id(pspace)
    h = (-t) * hopping + J * heis - (mu / 4) * (num ⊗ unit + unit ⊗ num)
    return nearest_neighbour_hamiltonian(fill(pspace, size(lattice)), h)
end

#
## Additional models
#

"""
    j1_j2_model([T::Type{T}, symm::Type{S},] lattice::InfiniteSquare;
                J1=1.0, J2=1.0, spin=1//2, sublattice=true)

Square lattice ``J_1\\text{-}J_2`` model, defined by the Hamiltonian

```math
H = J_1 \\sum_{\\langle i,j \\rangle} \\vec{S}_i \\cdot \\vec{S}_j
+ J_2 \\sum_{\\langle\\langle i,j \\rangle\\rangle} \\vec{S}_i \\cdot \\vec{S}_j,
```

where ``\\vec{S}_i = (S_i^x, S_i^y, S_i^z)``. We denote the nearest and next-nearest neighbor
terms using ``\\langle i,j \\rangle`` and ``\\langle\\langle i,j \\rangle\\rangle``,
respectively. The `sublattice` kwarg enables a single-site unit cell ground state via a
unitary sublattice rotation.
"""
function j1_j2_model(lattice::InfiniteSquare; kwargs...)
    return j1_j2_model(ComplexF64, Trivial, lattice; kwargs...)
end
function j1_j2_model(
        T::Type{<:Number}, S::Type{<:Sector}, lattice::InfiniteSquare;
        J1 = 1.0, J2 = 1.0, spin = 1 // 2, sublattice = true,
    )
    term_AA = SO.S_exchange(T, S; spin)
    term_AB = if sublattice
        -SO.S_x_S_x(T, S; spin) + SO.S_y_S_y(T, S; spin) - SO.S_z_S_z(T, S; spin)  # Apply sublattice rotation
    else
        term_AA
    end
    spaces = fill(domain(term_AA)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => J1 * term_AB for neighbor in nearest_neighbours(lattice))...,
        (neighbor => J2 * term_AA for neighbor in next_nearest_neighbours(lattice))...,
    )
end

"""
    pwave_superconductor([T=ComplexF64,] lattice::InfiniteSquare; t=1, μ=2, Δ=1)

Square lattice ``(p + ip)``-wave superconductor model, defined by the Hamiltonian

```math
    H = -\\sum_{\\langle i,j \\rangle} \\left( t c_i^\\dagger c_j +
    \\Delta_{ij} c_i c_j + \\text{h.c.} \\right) - \\mu \\sum_i n_i,
```

where ``t`` is the hopping amplitude, ``\\Delta_{ij}`` specifies the superconducting gap, ``\\mu``
is the chemical potential, and ``n_i = c_i^\\dagger c_i`` is the fermionic number operator.
For ``p + ip``-wave, ``\\Delta_{ij} = \\Delta`` on horizontal bonds,
and ``i \\Delta`` on vertical bonds.
"""
function pwave_superconductor(lattice::InfiniteSquare; kwargs...)
    return pwave_superconductor(ComplexF64, lattice; kwargs...)
end
function pwave_superconductor(
        T::Type{<:Number}, lattice::InfiniteSquare;
        t::Number = 1, μ::Number = 2, Δ::Number = 1
    )
    physical_space = FO.fermion_space(Trivial)
    spaces = fill(physical_space, (lattice.Nrows, lattice.Ncols))
    # hopping and pairing operators
    hopp = -t * FO.f_hopping(T, Trivial)
    pair = FO.f_min_f_min(T, Trivial)
    pair_x, pair_y = Δ * pair, im * Δ * pair

    # on-site
    h0 = -μ * FO.f_num(T, Trivial)
    # two-site (x-direction)
    hx = hopp + (pair_x + pair_x')
    # two-site (y-direction)
    hx = hopp + (pair_y + pair_y')

    x_neighbors = filter(n -> n[2].I[2] > n[1].I[2], nearest_neighbours(lattice))
    y_neighbors = filter(n -> n[2].I[1] > n[1].I[1], nearest_neighbours(lattice))
    return LocalOperator(
        spaces,
        ([idx] => h0 for idx in vertices(lattice))...,
        (neighbor => hx for neighbor in x_neighbors)...,
        (neighbor => hy for neighbor in y_neighbors)...,
    )
end
