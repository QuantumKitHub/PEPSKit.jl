"""
$(TYPEDEF)

Abstract super type for the collection of
Trotter evolution MPOs acting on 3 or more sites.
"""
abstract type TrotterMPOs end

Base.getindex(gate::TrotterMPOs, args...) = Base.getindex(gate.data, args...)

"""
    struct TrotterMPOs2ndNeighbor{T}

Collection of all Trotter evolution MPOs obtained from
a Hamiltonian containing up to 2nd neighbor terms
```
    H = ∑ᵢⱼ(┘ᵢⱼ + ┐ᵢⱼ + ┌ᵢⱼ + └ᵢⱼ)
```
where `┘`, `┐`, `┌`, `└` refer to the following 3-site clusters 
```
        3   3---2   2---1   1
        |       |   |       |
    1---2       1   3       2---3
```
`data[d][i, j]` is the `┘ᵢⱼ` MPO acting on the `[i, j]` southeast
cluster after the network is left-rotated by `90 x (d - 1)` degrees.
"""
struct TrotterMPOs2ndNeighbor{T} <: TrotterMPOs
    data::T
end

function TrotterMPOs2ndNeighbor(H::LocalOperator, dt::Number)
    return TrotterMPOs2ndNeighbor(
        [
            _get_gatempos_se(H, dt),
            _get_gatempos_se(rotl90(H), dt),
            _get_gatempos_se(rot180(H), dt),
            _get_gatempos_se(rotr90(H), dt),
        ]
    )
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
    _get_se3site_term(ham::LocalOperator, row::Int, col::Int)

Construct the term acting on the southeast 3-site cluster in `ham`.
```
    r-1     3
            ↓
    r   1-←-2
        c   c+1
```
"""
function _get_se3site_term(ham::LocalOperator, row::Int, col::Int)
    Nr, Nc = size(ham)
    T = scalartype(ham)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    sites = [
        CartesianIndex(row, col),
        CartesianIndex(row, col + 1),
        CartesianIndex(row - 1, col + 1),
    ]
    ss = map(sites) do site
        _get_site_term(ham, site)
    end
    nb1x = _get_bond_term(ham, (sites[1], sites[2]))
    nb1y = _get_bond_term(ham, (sites[2], sites[3]))
    nb2 = _get_bond_term(ham, (sites[1], sites[3]))
    # identity operator at each site
    units = map(sites) do site
        site_ = CartesianIndex(mod1(site[1], Nr), mod1(site[2], Nc))
        return id(T, physicalspace(ham)[site_])
    end
    # When iterating through ┘, └, ┌, ┐ clusters in the unit cell,
    # each site / NN-bond / NNN-bond is counted 12 / 4 / 2 times, respectively.
    term_site = (
        ss[1] ⊗ units[2] ⊗ units[3] +
            units[1] ⊗ ss[2] ⊗ units[3] +
            units[1] ⊗ units[2] ⊗ ss[3]
    ) / 12
    @tensor term_nb1[i' j' k'; i j k] :=
        (nb1x[i' j'; i j] * units[3][k' k] + units[1][i'; i] * nb1y[j' k'; j k]) / 4
    @tensor term_nb2[i' j' k'; i j k] := (nb2[i' k'; i k] * units[2][j'; j]) / 2
    return term_site + term_nb1 + term_nb2
end


"""
Obtain 3-site gate MPOs on southeast cluster at all positions `[row, col]`
```
    r-1        g3
                |
                ↓
    r   g1 -←- g2
        c      c+1
```
"""
function _get_gatempos_se(ham::LocalOperator, dt::Number)
    Nr, Nc = size(ham.lattice)
    return map(Iterators.product(1:Nr, 1:Nc)) do (row, col)
        term = _get_se3site_term(ham, row, col)
        return gate_to_mpo3(exp(-dt * term))
    end
end
