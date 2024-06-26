## Model Hamiltonians
# -------------------

"""
    square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)

Square lattice Heisenberg model.
By default, this implements a single site unit cell via a sublattice rotation.
"""
function square_lattice_heisenberg(
    ::Type{T}=ComplexF64; Jx=-1, Jy=1, Jz=-1
) where {T<:Number}
    physical_space = ComplexSpace(2)
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return NLocalOperator{NearestNeighbor}(H / 4)
end

"""
    square_lattice_pwave(; t=1, μ=2, Δ=1)

Square lattice p-wave superconductor model.
"""
function square_lattice_pwave(
    ::Type{T}=ComplexF64; t::Number=1, μ::Number=2, Δ::Number=1
) where {T<:Number}
    physical_space = Vect[FermionParity](0 => 1, 1 => 1)
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

    return AnisotropicNNOperator(h0, hx, hy)
end
