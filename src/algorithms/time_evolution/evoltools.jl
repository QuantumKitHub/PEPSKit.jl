const InfiniteState = Union{InfinitePEPS, InfinitePEPO}

function MPSKit.infinite_temperature_density_matrix(H::LocalOperator)
    T = scalartype(H)
    A = map(physicalspace(H)) do Vp
        ψ = permute(TensorKit.id(T, Vp), (1, 2))
        Vv = oneunit(Vp) # trivial (1D) virtual space
        virt = ones(T, domain(ψ) ← Vv ⊗ Vv ⊗ Vv' ⊗ Vv')
        return ψ * virt
    end
    return InfinitePEPO(cat(A; dims = 3))
end

"""
    get_expham(H::LocalOperator, dt::Number)

Compute `exp(-dt * op)` for each term `op` in `H`,
and combine them into a new LocalOperator.
Each `op` in `H` must be a single `TensorMap`.
"""
function get_expham(H::LocalOperator, dt::Number)
    return LocalOperator(
        physicalspace(H), (sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
end

"""
    is_nearest_neighbour(H::LocalOperator)

Check if an operator `H` contains only nearest neighbor terms.
"""
function is_nearest_neighbour(H::LocalOperator)
    return all(H.terms) do (sites, op)
        return numin(op) == 2 && sum(abs, Tuple(sites[2] - sites[1])) == 1
    end
end

"""
    is_equivalent_bond(bond1::NTuple{2,CartesianIndex{2}}, bond2::NTuple{2,CartesianIndex{2}}, (Nrow, Ncol)::NTuple{2,Int})

Check if two 2-site bonds are related by a (periodic) lattice translation.
"""
function is_equivalent_bond(
        bond1::NTuple{2, CartesianIndex{2}}, bond2::NTuple{2, CartesianIndex{2}},
        (Nrow, Ncol)::NTuple{2, Int},
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
function get_gateterm(gate::LocalOperator, bond::NTuple{2, CartesianIndex{2}})
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
$(SIGNATURES)

Use QR decomposition on two tensors `A`, `B` connected by a bond to get the reduced tensors.
When `A`, `B` are PEPSTensors,
```
        2                   1                                   1
        |                   |                                   |
    5 -A/B- 3   ====>   4 - X ← 2   1 ← a - 3   1 - b → 3   4 → Y - 2
        | ↘                 |            ↘           ↘          |
        4   1               3             2           2         3
```
When `A`, `B` are PEPOTensors, 
- If `gate_ax = 1`
```
    2   3                1  2                                1  2
      ↘ |                 ↘ |                                 ↘ |
    6 -A/B- 4   ====>   5 - X ← 3   1 ← a - 3   1 - b → 3   5 → Y - 3
        | ↘                 |            ↘           ↘          |
        5   1               4             2           2         4
```
- If `gate_ax = 2`
```
    2   3                   2         2           2             2
      ↘ |                   |          ↘           ↘            |
    6 -A/B- 4   ====>   5 - X ← 3   1 ← a - 3   1 - b → 3   5 → Y - 3
        | ↘                 | ↘                                 | ↘
        5   1               4  1                                4  1
```
"""
function _qr_bond(A::PT, B::PT; gate_ax::Int = 1) where {PT <: Union{PEPSTensor, PEPOTensor}}
    @assert 1 <= gate_ax <= numout(A)
    permA, permB, permX, permY = if A isa PEPSTensor
        ((2, 4, 5), (1, 3)), ((2, 3, 4), (1, 5)), (1, 4, 2, 3), Tuple(1:4)
    else
        if gate_ax == 1
            ((2, 3, 5, 6), (1, 4)), ((2, 3, 4, 5), (1, 6)), (1, 2, 5, 3, 4), Tuple(1:5)
        else
            ((1, 3, 5, 6), (2, 4)), ((1, 3, 4, 5), (2, 6)), (1, 2, 5, 3, 4), Tuple(1:5)
        end
    end
    X, a = left_orth(permute(A, permA))
    Y, b = left_orth(permute(B, permB))
    # no longer needed after TensorKit 0.15
    # @assert !isdual(space(a, 1))
    # @assert !isdual(space(b, 1))
    X, Y = permute(X, permX), permute(Y, permY)
    b = permute(b, ((3, 2), (1,)))
    return X, a, b, Y
end

"""
$(SIGNATURES)

Reconstruct the tensors connected by a bond from their `_qr_bond` results.
For PEPSTensors,
```
        -2                             -2
        |                               |
    -5- X - 1 - a - -3     -5 - b - 1 - Y - -3
        |        ↘               ↘      |
        -4        -1              -1   -4
```
For PEPOTensors
```
    -2  -3                          -2  -3
      ↘ |                             ↘ |
    -6- X - 1 - a - -4     -6 - b - 1 - Y - -4
        |        ↘               ↘      |
        -5        -1              -1   -5

        -3   -2              -2        -3
        |      ↘               ↘        |
    -6- X - 1 - a - -4     -6 - b - 1 - Y - -4
        | ↘                             | ↘
        -5 -1                          -5  -1
```
"""
function _qr_bond_undo(X::PEPSOrth, a::AbstractTensorMap, b::AbstractTensorMap, Y::PEPSOrth)
    @tensor A[-1; -2 -3 -4 -5] := X[-2 1 -4 -5] * a[1 -1 -3]
    @tensor B[-1; -2 -3 -4 -5] := b[-5 -1 1] * Y[-2 -3 -4 1]
    return A, B
end
function _qr_bond_undo(X::PEPOOrth, a::AbstractTensorMap, b::AbstractTensorMap, Y::PEPOOrth)
    if !isdual(space(a, 2))
        @tensor A[-1 -2; -3 -4 -5 -6] := X[-2 -3 1 -5 -6] * a[1 -1 -4]
        @tensor B[-1 -2; -3 -4 -5 -6] := b[-6 -1 1] * Y[-2 -3 -4 -5 1]
    else
        @tensor A[-1 -2; -3 -4 -5 -6] := X[-1 -3 1 -5 -6] * a[1 -2 -4]
        @tensor B[-1 -2; -3 -4 -5 -6] := b[-6 -2 1] * Y[-1 -3 -4 -5 1]
    end
    return A, B
end

"""
$(SIGNATURES)

Apply 2-site `gate` on the reduced matrices `a`, `b`
```
    -1← a --- 3 --- b ← -4          -2         -3
        ↓           ↓               ↓           ↓
        1           2               |----gate---|
        ↓           ↓       or      ↓           ↓
        |----gate---|               1           2
        ↓           ↓               ↓           ↓
        -2         -3           -1← a --- 3 --- b ← -4
```
"""
function _apply_gate(
        a::AbstractTensorMap, b::AbstractTensorMap,
        gate::AbstractTensorMap{T, S, 2, 2}, trscheme::TruncationStrategy
    ) where {T <: Number, S <: ElementarySpace}
    V = space(b, 1)
    need_flip = isdual(V)
    if isdual(space(a, 2))
        @tensor a2b2[-1 -2; -3 -4] := gate[1 2; -2 -3] * a[-1 1 3] * b[3 2 -4]
    else
        @tensor a2b2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * a[-1 1 3] * b[3 2 -4]
    end
    trunc = if trscheme isa FixedSpaceTruncation
        need_flip ? truncspace(flip(V)) : truncspace(V)
    else
        trscheme
    end

    # TODO: replace this with actual truncation error once TensorKit is updated
    ac, sc, bc = svd_compact!(a2b2; alg = LAPACK_QRIteration())
    a, s, b, ϵ = _truncate_compact((ac, sc, bc), trunc)

    a, b = absorb_s(a, s, b)
    if need_flip
        a, s, b = flip_svd(a, s, b)
    end
    return a, s, b, ϵ
end

"""
Convert a 3-site gate to MPO form by SVD, 
in which the axes are ordered as
```
    2               3               3
    ↓               ↓               ↓
    g1 ←- 3    1 ←- g2 ←- 4    1 ←- g3
    ↓               ↓               ↓
    1               2               2
```
"""
function gate_to_mpo3(
        gate::AbstractTensorMap{T, S, 3, 3}, trunc = trunctol(; atol = MPSKit.Defaults.tol)
    ) where {T <: Number, S <: ElementarySpace}
    Os = MPSKit.decompose_localmpo(MPSKit.add_util_leg(gate), trunc)
    g1 = removeunit(Os[1], 1)
    g2 = Os[2]
    g3 = removeunit(Os[3], 4)
    return [g1, g2, g3]
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
function _get_gatempo_se(ham::LocalOperator, dt::Number, row::Int, col::Int)
    Nr, Nc = size(ham)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    sites = [
        CartesianIndex(row, col),
        CartesianIndex(row, col + 1),
        CartesianIndex(row - 1, col + 1),
    ]
    nb1x = get_gateterm(ham, (sites[1], sites[2]))
    nb1y = get_gateterm(ham, (sites[2], sites[3]))
    nb2 = get_gateterm(ham, (sites[1], sites[3]))
    # identity operator at each site
    units = map(sites) do site
        site_ = CartesianIndex(mod1(site[1], Nr), mod1(site[2], Nc))
        return id(physicalspace(ham)[site_])
    end
    # when iterating through ┘, └, ┌, ┐ clusters in the unit cell,
    # NN / NNN bonds are counted 4 / 2 times, respectively.
    @tensor Odt[i' j' k'; i j k] :=
        -dt * (
        (nb1x[i' j'; i j] * units[3][k' k] + units[1][i'; i] * nb1y[j' k'; j k]) / 4 +
            (nb2[i' k'; i k] * units[2][j'; j]) / 2
    )
    op = exp(Odt)
    return gate_to_mpo3(op)
end

"""
Construct the 3-site gate MPOs on the southeast cluster 
for 3-site simple update on square lattice.
"""
function _get_gatempos_se(ham::LocalOperator, dt::Number)
    Nr, Nc = size(ham.lattice)
    return collect(_get_gatempo_se(ham, dt, r, c) for r in 1:Nr, c in 1:Nc)
end
