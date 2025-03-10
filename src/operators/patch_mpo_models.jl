"""
# PatchMPO

`PatchMPO` provides functionality for evaluating energy on a 2×2 patch of a 2D lattice.
This approach allows efficient computation of local energy contributions by mapping 
the 2D patch to a 1D chain with appropriate interactions. The current version supports
uniform interactions within the unit-cell (Note we don't require uniform PEPS which can take
m*n size). Extention to longer range interactions is not yet implemented.

## Patch Configuration

The 2×2 patch is mapped to a 4-site chain as follows:
```
O1--------O2          α1    α2    α3    α4
 |        |         a |   b |   c |  d  |   a
 |        |    =>   --O1----O2----O3----O4--
 |        |           |     |     |     |
O4--------O3          β1    β2    β3    β4
```

where O1, O2, O3, O4 represent the four sites and their corresponding operators.

## Usage Workflow

1. **Construct the MPO** for the 2×2 patch interactions:
    This is done by SVD in the current file.

2. **Modify PEPSSandwich tensors** to incorporate operators at each site:
   ```julia
   for (site, operator) in enumerate(mpo_operators)
       modified_A[site] = apply_operator_to_sandwich(A, site, operator)
   end
   ```
   Each modified tensor incorporates the expectation value ⟨ket|Oᵢ|bra⟩ and has an
   additional dimension corresponding to the MPO virtual bond dimension.

3. **Perform corner expansion** using the modified tensors:
   ```julia
   for corner in corners
       expanded_corners[corner] = expand_corner(corner_environments, modified_A)
   end
   ```

4. **Contract the expanded corners** to obtain the energy expectation value:
   ```julia
   energy = contract_patch(expanded_corners[1], expanded_corners[2],
                          expanded_corners[3], expanded_corners[4])
   ```

This method maintains the same computational complexity as direct contraction of individual
terms, but offers improved organization and reusability for complex Hamiltonians.
"""

## Constructors
const PatchMPO{T1<:AbstractTensorMap,T2<:AbstractTensorMap,T3<:AbstractTensorMap,T4<:AbstractTensorMap} = Tuple{
    T1,T2,T3,T4
}

"""
    decompose_patch_to_mpo(Ham::AbstractTensorMap; tol=eps(eltype(norm(Ham)))^(3 / 4)) -> PatchMPO

Decompose a four-site Hamiltonian into four single-site operators using sequential truncated SVD.
This transforms a 2×2 patch operator into an MPO representation for efficient contraction.

# Parameters
- `Ham`: Four-site operator tensor with shape [α1,α2,α3,α4; β1,β2,β3,β4]
- `tol`: Optional truncation tolerance. If not provided, will be calculated automatically.

# Returns
- `PatchMPO`: Tuple of four operators (OP1, OP2, OP3, OP4) representing the decomposed MPO
"""
function decompose_patch_to_mpo(Ham; tol=eps(eltype(norm(Ham)))^(3 / 4))
    tol *= norm(Ham)

    # Step 1: Separate first site (OP1)
    a = ones(oneunit(spacetype(Ham))) #auxiliary indices
    a′ = ones(oneunit(spacetype(Ham))')
    @tensor h1[a, α1, β1; α2, β2, α3, β3, α4, β4, a′] :=
        Ham[α1, α2, α3, α4; β1, β2, β3, β4] * a[a] * a′[a′]
    u1, s1, v1 = tsvd!(h1; trunc=truncerr(tol))
    us1 = u1 * sqrt(s1)
    vs1 = sqrt(s1) * v1
    @tensor OP1[a, α1; β1, b] := us1[a, α1, β1; b]

    # Step 2: Separate second site (OP2)
    @tensor h2[b, α2, β2; α3, β3, α4, β4, a] := vs1[b, ; α2, β2, α3, β3, α4, β4, a]
    u2, s2, v2 = tsvd!(h2; trunc=truncerr(tol))
    us2 = u2 * sqrt(s2)
    vs2 = sqrt(s2) * v2
    @tensor OP2[b, α2; β2, c] := us2[b, α2, β2; c]

    # Step 3: Separate third site (OP3)
    @tensor h3[c, α3, β3; α4, β4, a] := vs2[c, ; α3, β3, α4, β4, a]
    u3, s3, v3 = tsvd!(h3; trunc=truncerr(tol))
    us3 = u3 * sqrt(s3)
    vs3 = sqrt(s3) * v3
    @tensor OP3[c, α3; β3, d] := us3[c, α3, β3; d]

    # Step 4: Extract final site (OP4)
    @tensor OP4[d, α4; β4, a] := vs3[d, ; α4, β4, a]

    return (OP1, OP2, OP3, OP4)
end

"""
    embed_operator_in_patch(op::AbstractTensorMap, site_index::Tuple) -> AbstractTensorMap

Embed an operator into a specific location within a 2×2 patch's Hilbert space.

This function takes an operator that acts on 1-4 sites and embeds it into the full 
4-site space corresponding to a 2×2 patch, placing the operator at the specified sites.

# Arguments
- `op::AbstractTensorMap`: The operator to be embedded (1-4 site operator)
- `site_index::Tuple`: Specifies which sites the operator acts on:
  * `(i,)`: Single-site operator acting on site `i`
  * `(i,j)`: Two-site operator acting on the bond between sites `i` and `j`
  * `(i,j,k)`: Three-site operator acting on sites `i`, `j`, and `k`
  * `(i,j,k,l)`: Four-site operator acting on all four sites

# Returns
- The resulting operator embedded in the full 4-site Hilbert space

# Example
```julia
# Place Pauli X operator on site 1
op_x = σˣ(ComplexF64, Trivial)
full_op = embed_operator_in_patch(op_x, (1,))

# Place ZZ coupling between sites 1 and 2
op_zz = σᶻᶻ(ComplexF64, Trivial)
full_op = embed_operator_in_patch(op_zz, (1,2))
```
"""
function embed_operator_in_patch(op::AbstractTensorMap, site_index::Tuple)
    n = 4
    k = numout(op)
    k == length(site_index) || error("Input operator is not a $k-site operator.")
    Pspace = space(op, 1)
    id_op = id(Pspace^(n - k))

    if k == n
        ham = op
    elseif k == 1
        @tensor ham[α1, β1, α2, α3, α4, β2, β3, β4] :=
            op[α1; β1] * id_op[α2, α3, α4; β2, β3, β4]
    elseif k == 2
        @tensor ham[α1, α2, β1, β2, α3, α4, β3, β4] :=
            op[α1, α2; β1, β2] * id_op[α3, α4; β3, β4]
    elseif k == 3
        @tensor ham[α1, α2, α3, β1, β2, β3, α4, β4] :=
            op[α1, α2, α3; β1, β2, β3] * id_op[α4; β4]
    else
        error("Operator acting on $k sites is not supported.")
    end

    return permute(ham, index_perm(n, site_index))
end
function index_perm(n, index)
    idx = vcat(collect(index), collect(index .+ n))
    new_order = vcat(idx, filter(x -> !(x in idx), 1:(2 * n)))

    perm = zeros(Int, 2 * n)
    for (new_pos, val) in enumerate(new_order)
        perm[val] = new_pos
    end

    return Tuple(perm[1:n]), Tuple(perm[(n + 1):end])
end

# Model examples
# -------------------------
"""
    patch_mpo_transverse_field_ising([T=ComplexF64], [S=Trivial]; J=1.0, g=1.0) -> PatchMPO

Create a patch MPO for the transverse field Ising model.

# Parameters
- `T`: Number type for the tensors (default: ComplexF64)
- `S`: Symmetry sector type (default: Trivial)
- `J`: Strength of the ZZ coupling (default: 1.0)
- `g`: Strength of the transverse field (default: 1.0)

# Returns
- A tuple of four operators representing the decomposed patch MPO
"""
function patch_mpo_transverse_field_ising(; kwargs...)
    return patch_mpo_transverse_field_ising(ComplexF64, Trivial; kwargs...)
end
function patch_mpo_transverse_field_ising(
    T::Type{<:Number}, S::Union{Type{Trivial},Type{Z2Irrep}}; J=1.0, g=1.0
)
    ZZ = rmul!(σᶻᶻ(T, S), -J)
    X = rmul!(σˣ(T, S), g * -J)
    ham =
        sum(embed_operator_in_patch(X, (i,)) for i in 1:4) / 4.0 +
        sum(embed_operator_in_patch(ZZ, (i, i % 4 + 1)) for i in 1:4) / 2.0

    return decompose_patch_to_mpo(ham)
end

"""
    patch_mpo_heisenberg_XYZ([T=ComplexF64], S::Type{<:Sector}; 
                           Jx=-1.0, Jy=1.0, Jz=-1.0, spin=1//2) -> PatchMPO

Create a patch MPO for the Heisenberg XYZ model.

# Parameters
- `T`: Number type for the tensors (default: ComplexF64)
- `S`: Symmetry sector type (default: Trivial)
- `Jx`: X-coupling strength (default: -1.0)
- `Jy`: Y-coupling strength (default: 1.0)
- `Jz`: Z-coupling strength (default: -1.0)
- `spin`: Spin value (default: 1/2)

# Returns
- A tuple of four operators representing the decomposed patch MPO
"""
function patch_mpo_heisenberg_XYZ(; kwargs...)
    return patch_mpo_heisenberg_XYZ(ComplexF64, Trivial; kwargs...)
end
function patch_mpo_heisenberg_XYZ(
    T::Type{<:Number}, S::Type{<:Sector}; Jx=-1.0, Jy=1.0, Jz=-1.0, spin=1//2
)
    term =
        rmul!(S_xx(T, S; spin=spin), Jx) +
        rmul!(S_yy(T, S; spin=spin), Jy) +
        rmul!(S_zz(T, S; spin=spin), Jz)
    ham = sum(embed_operator_in_patch(term, (i, i % 4 + 1)) for i in 1:4) / 2.0

    return decompose_patch_to_mpo(ham)
end

"""
    patch_mpo_j1_j2([T=ComplexF64], S::Type{<:Sector}; 
                  J1=1.0, J2=1.0, spin=1//2, sublattice=true) -> PatchMPO

Create a patch MPO for the J1-J2 Heisenberg model.

# Parameters
- `T`: Number type for the tensors
- `S`: Symmetry sector type
- `J1`: Nearest neighbor coupling strength (default: 1.0)
- `J2`: Next-nearest neighbor coupling strength (default: 1.0)
- `spin`: Spin value (default: 1/2)
- `sublattice`: Whether to apply sublattice rotation (default: true)

# Returns
- A tuple of four operators representing the decomposed patch MPO
"""
function patch_mpo_j1_j2(; kwargs...)
    return patch_mpo_j1_j2(ComplexF64, Trivial; kwargs...)
end
function patch_mpo_j1_j2(
    T::Type{<:Number}, S::Type{<:Sector}; J1=1.0, J2=1.0, spin=1//2, sublattice=true
)
    term_AA = S_xx(T, S; spin) + S_yy(T, S; spin) + S_zz(T, S; spin)
    term_AB = if sublattice
        -S_xx(T, S; spin) + S_yy(T, S; spin) - S_zz(T, S; spin)  # Apply sublattice rotation
    else
        term_AA
    end

    term_AA *= J2
    term_AB *= J1
    ham =
        sum(embed_operator_in_patch(term_AB, (i, i % 4 + 1)) for i in 1:4) / 2.0 +
        sum(embed_operator_in_patch(term_AA, (i, (i + 1) % 4 + 1)) for i in 1:2:4)
    return decompose_patch_to_mpo(ham)
end

"""
    patch_mpo_pwave_superconductor([T=ComplexF64]; t=1, μ=2, Δ=1) -> PatchMPO

Create a patch MPO for a square lattice p-wave superconductor model.

# Parameters
- `T`: Number type for the tensors (default: ComplexF64)
- `t`: Hopping parameter (default: 1)
- `μ`: Chemical potential (default: 2)
- `Δ`: Superconducting gap (default: 1)

# Returns
- A tuple of four operators representing the decomposed patch MPO
"""
function patch_mpo_pwave_superconductor(
    T::Type{<:Number}=ComplexF64; t::Number=1, μ::Number=2, Δ::Number=1
)
    physical_space = Vect[FermionParity](0 => 1, 1 => 1)

    # on-site
    h0 = zeros(T, physical_space ← physical_space)
    block(h0, FermionParity(1)) .= -μ

    # two-site (x-direction)
    hx = zeros(T, physical_space^2 ← physical_space^2)
    block(hx, FermionParity(0)) .= [0 -Δ; -Δ 0]
    block(hx, FermionParity(1)) .= [0 -t; -t 0]

    # two-site (y-direction)
    hy = zeros(T, physical_space^2 ← physical_space^2)
    block(hy, FermionParity(0)) .= [0 Δ*im; -Δ*im 0]
    block(hy, FermionParity(1)) .= [0 -t; -t 0]

    ham =
        sum(embed_operator_in_patch(h0, (i,)) for i in 1:4) / 4.0 +
        sum(embed_operator_in_patch(hx, (i, i % 4 + 1)) for i in 1:2:4) / 2.0 +
        sum(embed_operator_in_patch(hy, (i, i % 4 + 1)) for i in 2:2:4) / 2.0
    return decompose_patch_to_mpo(ham)
end

"""
    patch_mpo_hubbard_model(T::Type{<:Number}, particle_symmetry::Type{<:Sector},
                          spin_symmetry::Type{<:Sector};
                          t=1.0, U=1.0, mu=0.0, n=0) -> PatchMPO

Create a patch MPO for the Hubbard model.

# Parameters
- `T`: Number type for the tensors
- `particle_symmetry`: Symmetry sector type for particles
- `spin_symmetry`: Symmetry sector type for spins
- `t`: Hopping parameter (default: 1.0)
- `U`: On-site interaction strength (default: 1.0)
- `mu`: Chemical potential (default: 0.0)
- `n`: Fixed particle number (default: 0, currently only n=0 is supported)

# Returns
- A tuple of four operators representing the decomposed patch MPO

# Throws
- `AssertionError`: If a fixed particle number other than 0 is specified
"""
function patch_mpo_hubbard_model(
    T::Type{<:Number},
    particle_symmetry::Type{<:Sector},
    spin_symmetry::Type{<:Sector};
    t=1.0,
    U=1.0,
    mu=0.0,
    n::Integer=0,
)
    @assert n == 0 "Currently no support for imposing a fixed particle number"
    N = MPSKitModels.e_number(T, particle_symmetry, spin_symmetry)
    pspace = space(N, 1)
    unit = TensorKit.id(pspace)
    hopping =
        MPSKitModels.e⁺e⁻(T, particle_symmetry, spin_symmetry) +
        MPSKitModels.e⁻e⁺(T, particle_symmetry, spin_symmetry)
    interaction_term = MPSKitModels.nꜛnꜜ(T, particle_symmetry, spin_symmetry)
    site_term = U * interaction_term - mu * N
    h = (-t) * hopping + (1 / 4) * (site_term ⊗ unit + unit ⊗ site_term)

    ham = sum(embed_operator_in_patch(h, (i, i % 4 + 1)) for i in 1:4) / 2.0
    return decompose_patch_to_mpo(ham)
end

"""
    patch_mpo_tj_model(T::Type{<:Number}, particle_symmetry::Type{<:Sector},
                     spin_symmetry::Type{<:Sector};
                     t=2.5, J=1.0, mu=0.0, slave_fermion=false) -> PatchMPO

Create a patch MPO for the t-J model.

# Parameters
- `T`: Number type for the tensors
- `particle_symmetry`: Symmetry sector type for particles
- `spin_symmetry`: Symmetry sector type for spins
- `t`: Hopping parameter (default: 2.5)
- `J`: Exchange coupling (default: 1.0)
- `mu`: Chemical potential (default: 0.0)
- `slave_fermion`: Whether to use slave fermion representation (default: false)

# Returns
- A tuple of four operators representing the decomposed patch MPO
"""
function patch_mpo_tj_model(
    T::Type{<:Number},
    particle_symmetry::Type{<:Sector},
    spin_symmetry::Type{<:Sector};
    t=2.5,
    J=1.0,
    mu=0.0,
    slave_fermion::Bool=false,
)
    hopping =
        TJOperators.e_plusmin(particle_symmetry, spin_symmetry; slave_fermion) +
        TJOperators.e_minplus(particle_symmetry, spin_symmetry; slave_fermion)
    num = TJOperators.e_number(particle_symmetry, spin_symmetry; slave_fermion)
    heis =
        TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) -
        (1 / 4) * (num ⊗ num)
    pspace = space(num, 1)
    unit = TensorKit.id(pspace)
    h = (-t) * hopping + J * heis - (mu / 4) * (num ⊗ unit + unit ⊗ num)
    if T <: Real
        h = real(h)
    end

    ham = sum(embed_operator_in_patch(h, (i, i % 4 + 1)) for i in 1:4) / 2.0
    return decompose_patch_to_mpo(ham)
end
