"""
    get_expham(dt::Float64, H::LocalOperator)

Compute `exp(-dt * H)` from Hamiltonian `H`.
Each term in `H` must be a single `TensorMap`.
"""
function get_expham(dt::Float64, H::LocalOperator)
    return LocalOperator(
        H.lattice, Tuple(sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
end

"""
    is_equivalent_bond(bond1::NTuple{2,CartesianIndex{2}}, bond2::NTuple{2,CartesianIndex{2}}, (Nrow, Ncol)::NTuple{2,Int})

Check if two 2-site bonds are related by a (periodic) lattice translation.
"""
function is_equivalent_bond(
    bond1::NTuple{2,CartesianIndex{2}},
    bond2::NTuple{2,CartesianIndex{2}},
    (Nrow, Ncol)::NTuple{2,Int},
)
    r1 = bond1[1] - bond1[2]
    r2 = bond2[1] - bond2[2]
    shift_row = bond1[1][1] - bond2[1][1]
    shift_col = bond1[1][2] - bond2[1][2]
    return r1 == r2 && mod(shift_row, Nrow) == 0 && mod(shift_col, Ncol) == 0
end

"""
    get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})

Get the term of a 2-site gate acting on a certain bond.
Input `gate` should only include one term for each nearest neighbor bond.
"""
function get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})
    bonds = findall(p -> is_equivalent_bond(p.first, bond, size(gate.lattice)), gate.terms)
    if length(bonds) == 0
        # try reversed site order
        bonds = findall(
            p -> is_equivalent_bond(p.first, reverse(bond), size(gate.lattice)), gate.terms
        )
        if length(bonds) == 1
            return permute(gate.terms[bonds[1]].second, ((2, 1), (4, 3)))
        elseif length(bonds) == 0
            # if term not found, return the zero operator
            dtype = scalartype(gate.terms[1].second)
            V = space(gate.terms[1].second, 1)
            return zeros(dtype, V ⊗ V ← V ⊗ V)
        else
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        end
    else
        (length(bonds) == 1) ||
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        return gate.terms[bonds[1]].second
    end
end

"""
Use QR decomposition on two tensors connected by a bond
to get the reduced tensors
```
        2                   1
        |                   |
    5 - A ← 3   ====>   4 - X ← 2   1 ← a ← 3
        | ↘                 |            ↘
        4   1               3             2

        2                               1
        |                               |
    5 ← B - 3   ====>   1 ← b → 3   4 → Y - 2
        | ↘                  ↘          |
        4   1                 2         3
```
"""
function _qr_bond(A::PEPSTensor, B::PEPSTensor)
    # TODO: relax dual requirement on the bonds
    @assert isdual(space(A, 3)) # currently only allow A ← B
    X, a = leftorth(A, ((2, 4, 5), (1, 3)))
    Y, b = leftorth(B, ((2, 3, 4), (1, 5)))
    @assert !isdual(space(a, 1))
    @assert !isdual(space(b, 1))
    X = permute(X, (1, 4, 2, 3))
    Y = permute(Y, (1, 2, 3, 4))
    b = permute(b, ((3, 2), (1,)))
    return X, a, b, Y
end

"""
Reconstruct the tensors connected by a bond from their QR results
obtained from `_qr_bond`
```
        -2                             -2
        |                               |
    -5- X - 1 - a - -3     -5 - b - 1 - Y - -3
        |        ↘               ↘      |
        -4        -1              -1   -4
```
"""
function _qr_bond_undo(X::PEPSOrth, a::AbstractTensorMap, b::AbstractTensorMap, Y::PEPSOrth)
    @tensor A[-1; -2 -3 -4 -5] := X[-2 1 -4 -5] * a[1 -1 -3]
    @tensor B[-1; -2 -3 -4 -5] := b[-5 -1 1] * Y[-2 -3 -4 1]
    return A, B
end

"""
Apply 2-site `gate` on the reduced matrices `a`, `b`
```
    -1← a -← 3 -← b ← -4
        ↓           ↓
        1           2
        ↓           ↓
        |----gate---|
        ↓           ↓
        -2         -3
```
"""
function _apply_gate(
    a::AbstractTensorMap{T,S},
    b::AbstractTensorMap{T,S},
    gate::AbstractTensorMap{T,S,2,2},
    trscheme::TruncationScheme,
) where {T<:Number,S<:ElementarySpace}
    @tensor a2b2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * a[-1 1 3] * b[3 2 -4]
    return tsvd!(
        a2b2;
        trunc=((trscheme isa FixedSpaceTruncation) ? truncspace(space(a, 3)) : trscheme),
        alg=TensorKit.SVD(),
    )
end

"""
Convert a 3-site gate to MPO form by SVD, 
in which the axes are ordered as
```
    1               2               2
    ↑               ↑               ↑
    g1 ←- 3    1 ←- g2 ←- 4    1 ←- g3
    ↑               ↑               ↑
    2               3               3
```
"""
function gate_to_mpo3(
    gate::AbstractTensorMap{T,S,3,3}; svderr=1e-12
) where {T<:Number,S<:ElementarySpace}
    g1, s, g23 = tsvd(gate, ((1, 4), (2, 5, 3, 6)); trunc=truncerr(svderr))
    g1, g23 = PEPSKit.absorb_s(g1, s, g23)
    g2, s, g3 = tsvd(g23, ((1, 2, 3), (4, 5)); trunc=truncerr(svderr))
    g2, g3 = PEPSKit.absorb_s(g2, s, g3)
    # for possibly better numerical stability
    g1.data[:] = round.(g1.data; digits=14)
    g2.data[:] = round.(g2.data; digits=14)
    g3.data[:] = round.(g3.data; digits=14)
    return [g1, g2, g3]
end

"""
Obtain the 3-site gate MPO on the southwest cluster at position `[row, col]`
```
    r-1 g1
        |       
        ↑
    r   g2 -←- g3
        c      c+1
```
"""
function _get_gatempo_sw(gate::LocalOperator, row::Int, col::Int)
    Nr, Nc = size(gate.lattice)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    unit = id(space(gate.terms[1].second, 1))
    sites = (
        CartesianIndex(row - 1, col), CartesianIndex(row, col), CartesianIndex(row, col + 1)
    )
    nb1y = get_gateterm(gate, (sites[1], sites[2]))
    nb1x = get_gateterm(gate, (sites[2], sites[3]))
    nb2 = get_gateterm(gate, (sites[1], sites[3]))
    op = (1 / 2) * (nb1y ⊗ unit + unit ⊗ nb1x) + permute(nb2 ⊗ unit, ((1, 3, 2), (4, 6, 5)))
    return gate_to_mpo3(op)
end

"""
Obtain the 3-site gate MPO on the southeast cluster at position `[row, col]`
```
    r-1        g3
                |
                ↓
    r   g1 -←- g2
        c      c+1
```
"""
function _get_gatempo_se(gate::LocalOperator, row::Int, col::Int)
    Nr, Nc = size(gate.lattice)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    unit = id(space(gate.terms[1].second, 1))
    sites = (
        CartesianIndex(row, col),
        CartesianIndex(row, col + 1),
        CartesianIndex(row - 1, col + 1),
    )
    nb1x = get_gateterm(gate, (sites[1], sites[2]))
    nb1y = get_gateterm(gate, (sites[2], sites[3]))
    nb2 = get_gateterm(gate, (sites[1], sites[3]))
    op = (1 / 2) * (nb1x ⊗ unit + unit ⊗ nb1y) + permute(nb2 ⊗ unit, ((1, 3, 2), (4, 6, 5)))
    return gate_to_mpo3(op)
end

"""
Construct the 3-site gate MPOs for simple update for a Hamiltonian 
that contains up to next nearest neighbor terms on square lattice.
"""
function _get_gatempos(gate::LocalOperator)
    Nr, Nc = size(gate.lattice)
    return Dict(
        :sw => collect(_get_gatempo_sw(gate, r, c) for r in 1:Nr, c in 1:Nc),
        :se => collect(_get_gatempo_se(gate, r, c) for r in 1:Nr, c in 1:Nc),
    )
end
