"""
    get_gate(dt::Float64, H::LocalOperator)

Compute `exp(-dt * H)` from the nearest neighbor Hamiltonian `H`.
"""
function get_gate(dt::Float64, H::LocalOperator)
    @assert all([
        numin(op) == 2 && norm(Tuple(terms[2] - terms[1])) == 1.0 for (terms, op) in H.terms
    ]) "Only nearest-neighbour terms allowed"
    return LocalOperator(
        H.lattice, Tuple(sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
end

"""
    is_equivalent(
        bond1::NTuple{2,CartesianIndex{2}},
        bond2::NTuple{2,CartesianIndex{2}},
        (Nrow, Ncol)::NTuple{2,Int},
    )


Check if two 2-site bonds are related by a (periodic) lattice translation.
"""
function is_equivalent(
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
    label = findall(p -> is_equivalent(p.first, bond, size(gate.lattice)), gate.terms)
    if length(label) == 0
        # try reversed site order
        label = findall(
            p -> is_equivalent(p.first, reverse(bond), size(gate.lattice)), gate.terms
        )
        @assert length(label) == 1
        return permute(gate.terms[label[1]].second, ((2, 1), (4, 3)))
    else
        @assert length(label) == 1
        return gate.terms[label[1]].second
    end
end

"""
$(SIGNATURES)

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
$(SIGNATURES)

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
$(SIGNATURES)

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
