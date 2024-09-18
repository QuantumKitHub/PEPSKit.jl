using MPSKitModels

## Model Hamiltonians
# -------------------

"""
    square_lattice_tf_ising(::Type{T}=ComplexF64; J=1, h=1, unitcell=(1, 1))

Square lattice transverse field Ising model.
"""
function square_lattice_tf_ising(
    elt::Type{T}=ComplexF64,
    symm::Type{<:Sector}=Trivial;
    J=1,
    h=1,
    unitcell::Tuple{Int,Int}=(1, 1),
) where {T<:Number}
    term_zz = rmul!(σᶻᶻ(elt, symm), -J)
    term_x = rmul!(σˣ(elt, symm), -J * h)
    lattice = fill(domain(term_x)[1], 1, 1)
    hzz = nearest_neighbour_hamiltonian(lattice, term_zz)
    return repeat(
        LocalOperator(lattice, hzz.terms..., (CartesianIndex(1, 1),) => term_x), unitcell...
    )
end

"""
    square_lattice_heisenberg(::Type{T}=ComplexF64; Jx=-1, Jy=1, Jz=-1, unitcell=(1, 1))

Square lattice Heisenberg model.
By default, this implements a single site unit cell via a sublattice rotation.
"""
function square_lattice_heisenberg(
    elt::Type{T}=ComplexF64,
    symm::Type{<:Sector}=Trivial;
    Jx=-1,
    Jy=1,
    Jz=-1,
    unitcell::Tuple{Int,Int}=(1, 1),
) where {T<:Number}
    term =
        rmul!(S_xx(elt, symm), Jx) + rmul!(S_yy(elt, symm), Jy) + rmul!(S_zz(elt, symm), Jz)
    lattice = fill(domain(term)[1], 1, 1)
    return repeat(nearest_neighbour_hamiltonian(lattice, term), unitcell...)
end

"""
    square_lattice_j1j2(::Type{T}=ComplexF64; J1=1, J2=1, unitcell=(1, 1), sublattice=true)


Square lattice J₁-J₂ model. The `sublattice` kwarg enables a single site unit cell via a
sublattice rotation.
"""
function square_lattice_j1j2(
    elt::Type{T}=ComplexF64,
    symm::Type{<:Sector}=Trivial;
    J1=1,
    J2=1,
    unitcell::Tuple{Int,Int}=(1, 1),
    sublattice=true,
) where {T<:Number}
    term_AA = S_xx(elt, symm) + S_yy(elt, symm) + S_zz(elt, symm)
    term_AB = sublattice ? -S_xx(elt, symm) + S_yy(elt, symm) - S_zz(elt, symm) : term_AA  # Apply sublattice rotation
    lattice = fill(domain(term_AA)[1], 1, 1)

    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        nearest_x = I + CartesianIndex(1, 0)
        nearest_y = I + CartesianIndex(0, 1)
        next_xy = I + CartesianIndex(1, 1)
        push!(terms, (I, nearest_x) => J1 * term_AB)
        push!(terms, (I, nearest_y) => J1 * term_AB)
        push!(terms, (I, next_xy) => J2 * term_AA)
        push!(terms, (nearest_x, nearest_y) => J2 * term_AA)
    end

    return repeat(LocalOperator(lattice, terms...), unitcell...)
end

"""
    square_lattice_pwave(::Type{T}=ComplexF64; t=1, μ=2, Δ=1, unitcell=(1, 1))

Square lattice p-wave superconductor model.
"""
function square_lattice_pwave(
    ::Type{T}=ComplexF64;
    t::Number=1,
    μ::Number=2,
    Δ::Number=1,
    unitcell::Tuple{Int,Int}=(1, 1),
) where {T<:Number}
    physical_space = Vect[FermionParity](0 => 1, 1 => 1)
    lattice = fill(physical_space, 1, 1)

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

    return repeat(
        LocalOperator(
            lattice,
            (CartesianIndex(1, 1),) => h0,
            (CartesianIndex(1, 1), CartesianIndex(1, 2)) => hx,
            (CartesianIndex(1, 1), CartesianIndex(2, 1)) => hy,
        ),
        unitcell...,
    )
end
