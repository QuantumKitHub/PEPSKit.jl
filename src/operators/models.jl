## Model Hamiltonians
# -------------------
function nearest_neighbour_hamiltonian(
    lattice::Matrix{S}, h::AbstractTensorMap{S,2,2}
) where {S}
    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        J1 = I + CartesianIndex(1, 0)
        J2 = I + CartesianIndex(0, 1)
        push!(terms, (I, J1) => h)
        push!(terms, (I, J2) => h)
    end
    return LocalOperator(lattice, terms...)
end

function MPSKitModels.transverse_field_ising(
    T::Type{<:Number},
    S::Union{Type{Trivial},Type{Z2Irrep}},
    lattice::InfiniteSquare=InfiniteSquare(1, 1);
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

function MPSKitModels.heisenberg_XYZ(
    lattice::InfiniteSquare=InfiniteSquare(1, 1); kwargs...
)
    return heisenberg_XYZ(ComplexF64, Trivial, lattice; kwargs...)
end
function MPSKitModels.heisenberg_XYZ(
    T::Type{<:Number},
    S::Type{<:Sector},
    lattice::InfiniteSquare;
    Jx=-1.0,
    Jy=1.0,
    Jz=-1.0,
    spin=1//2,
)
    term =
        rmul!(S_xx(T, S; spin=spin), Jx) +
        rmul!(S_yy(T, S; spin=spin), Jy) +
        rmul!(S_zz(T, S; spin=spin), Jz)
    spaces = fill(domain(term)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces, (neighbor => term for neighbor in nearest_neighbours(lattice))...
    )
end

"""
    j1_j2([elt::Type{T}], [symm::Type{S}], [lattice::InfiniteSquare];
                        J1=1.0, J2=1.0, spin=1//2, sublattice=true)

Square lattice J₁-J₂ model. The `sublattice` kwarg enables a single site unit cell via a
sublattice rotation.
"""
function j1_j2(
    T::Type{<:Number}=ComplexF64,
    S::Type{<:Sector}=Trivial,
    lattice::InfiniteSquare=InfiniteSquare(1, 1);
    J1=1.0,
    J2=1.0,
    spin=1//2,
    sublattice=true,
)
    term_AA = S_xx(T, S; spin) + S_yy(T, S; spin) + S_zz(T, S; spin)
    term_AB = if sublattice
        -S_xx(T, S; spin) + S_yy(T, S; spin) - S_zz(T, S; spin)  # Apply sublattice rotation
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
    pwave_superconductor(::Type{T}=ComplexF64; t=1, μ=2, Δ=1, unitcell=(1, 1))

Square lattice p-wave superconductor model.
"""
function pwave_superconductor(lattice::InfiniteSquare=InfiniteSquare(1, 1); kwargs...)
    return pwave_superconductor(ComplexF64, lattice; kwargs...)
end
function pwave_superconductor(
    T::Type{<:Number}, lattice::InfiniteSquare; t::Number=1, μ::Number=2, Δ::Number=1
)
    physical_space = Vect[FermionParity](0 => 1, 1 => 1)
    spaces = fill(physical_space, (lattice.Nrows, lattice.Ncols))

    # on-site
    h0 = TensorMap(zeros, T, physical_space ← physical_space)
    block(h0, FermionParity(1)) .= -μ

    # two-site (x-direction)
    hx = TensorMap(zeros, T, physical_space^2 ← physical_space^2)
    block(hx, FermionParity(0)) .= [0 -Δ; -Δ 0]
    block(hx, FermionParity(1)) .= [0 -t; -t 0]

    # two-site (y-direction)
    hy = TensorMap(zeros, T, physical_space^2 ← physical_space^2)
    block(hy, FermionParity(0)) .= [0 Δ*im; -Δ*im 0]
    block(hy, FermionParity(1)) .= [0 -t; -t 0]

    x_neighbors = filter(n -> n[2].I[2] > n[1].I[2], nearest_neighbours(lattice)) 
    y_neighbors = filter(n -> n[2].I[1] > n[1].I[1], nearest_neighbours(lattice)) 
    return LocalOperator(
        spaces,
        ((idx,) => h0 for idx in vertices(lattice))...,
        (neighbor => hx for neighbor in x_neighbors)...,
        (neighbor => hy for neighbor in y_neighbors)...,
    )
end
