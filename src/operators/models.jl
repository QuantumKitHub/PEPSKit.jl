## Model Hamiltonians
# -------------------

"""
    square_lattice_tf_ising(::Type{T}=ComplexF64; J=1, h=1, unitcell=(1, 1))

Square lattice transverse field Ising model.
"""
function square_lattice_tf_ising(
    ::Type{T}=ComplexF64; J=1, h=1, unitcell::Tuple{Int,Int}=(1, 1)
) where {T<:Number}
    physical_space = ComplexSpace(2)
    lattice = fill(physical_space, 1, 1)
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    hzz = nearest_neighbour_hamiltonian(lattice, -J * σz ⊗ σz)
    return repeat(
        LocalOperator(lattice, hzz.terms..., (CartesianIndex(1, 1),) => -J * h * σx),
        unitcell...,
    )
end

"""
    square_lattice_heisenberg(::Type{T}=ComplexF64; Jx=-1, Jy=1, Jz=-1, unitcell=(1, 1))

Square lattice Heisenberg model.
By default, this implements a single site unit cell via a sublattice rotation.
"""
function square_lattice_heisenberg(
    ::Type{T}=ComplexF64; Jx=-1, Jy=1, Jz=-1, unitcell::Tuple{Int,Int}=(1, 1)
) where {T<:Number}
    physical_space = ComplexSpace(2)
    lattice = fill(physical_space, 1, 1)
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return repeat(nearest_neighbour_hamiltonian(lattice, H / 4), unitcell...)
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
