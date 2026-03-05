"""
    struct TrotterMPOs{T <: Vector}

Collection of Trotter evolution MPOs obtained from
a Hamiltonian containing long-range or multi-site terms
"""
struct TrotterMPOs{T <: Vector}
    data::T
end

"""
Convert an N-site gate to MPO form by SVD, 
in which the axes are ordered as
```
    site 1      mid sites      site N
    2               3               3
    ↓               ↓               ↓
    g1 ←- 3    1 ←- g ←- 4    1 ←- gN
    ↓               ↓               ↓
    1               2               2
```
"""
function gate_to_mpo(
        gate::AbstractTensorMap{T, S, N, N}, trunc = trunctol(; atol = MPSKit.Defaults.tol)
    ) where {T <: Number, S <: ElementarySpace, N}
    Os = MPSKit.decompose_localmpo(MPSKit.add_util_leg(gate), trunc)
    return map(1:N) do i
        if i == 1
            return removeunit(Os[1], 1)
        elseif i == N
            return removeunit(Os[N], 4)
        else
            return Os[i]
        end
    end
end

# Specialized functions to trotterize Hamiltonians
# with long-range/multi-site terms

## Next-nearest neighbor H

"""
Trotterize a Hamiltonian containing up to 2nd neighbor terms.
```
    H = ∑ᵢⱼ(Γᵢⱼ + ⅂ᵢⱼ + ⅃ᵢⱼ + Lᵢⱼ)
```
where `Γ`, `⅂`, `⅃`, `L` refer to the following 3-site clusters 
```
    NORTHWEST   NORTHEAST
        2---1   3---2
        |           |
        3           1

        1           3
        |           |
        2---3   1---2
    SOUTHWEST   SOUTHEAST
```
acting on the the elemental square plaquette
with southwest corner at `[i, j]`.
"""
function trotterize_nnn(H::LocalOperator, dt::Number)
    Nr, Nc = size(H)
    # iterate through corner `d` in outermost loop
    terms = map(Iterators.product(1:Nr, 1:Nc, 1:4)) do (r, c, d)
        return _get_nnn_mpo(H, dt, d, r, c)
    end
    return TrotterMPOs(vec(terms))
end

"""
Get coordinates of sites in the 3-site triangular cluster
used in Trotter evolution with next-nearest neighbor gates,
with southwest corner at `[row, col]`.
"""
function _nnn_cluster_sites(dir::Int, row::Int, col::Int)
    @assert 1 <= dir <= 4
    return if dir == NORTHWEST
        [
            CartesianIndex(row - 1, col + 1),
            CartesianIndex(row - 1, col),
            CartesianIndex(row, col),
        ]
    elseif dir == NORTHEAST
        [
            CartesianIndex(row, col + 1),
            CartesianIndex(row - 1, col + 1),
            CartesianIndex(row - 1, col),
        ]
    elseif dir == SOUTHEAST
        [
            CartesianIndex(row, col),
            CartesianIndex(row, col + 1),
            CartesianIndex(row - 1, col + 1),
        ]
    else # dir == SOUTHWEST
        [
            CartesianIndex(row - 1, col),
            CartesianIndex(row, col),
            CartesianIndex(row, col + 1),
        ]
    end
end

"""
Construct the evolution MPO acting on the 3-site triangular cluster
in the square plaquette whose southwest corner is at `[row, col]`.
`dir` takes values between 1 (`NORTHWEST`) and 4 (`SOUTHWEST`).
"""
function _get_nnn_mpo(ham::LocalOperator, dt::Number, dir::Int, row::Int, col::Int)
    Nr, Nc = size(ham)
    T = scalartype(ham)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    sites = _nnn_cluster_sites(dir, row, col)
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
    # When iterating through all triangular clusters in the unit cell,
    # each site / NN-bond / NNN-bond is counted 12 / 4 / 2 times, respectively.
    term_site = (
        ss[1] ⊗ units[2] ⊗ units[3] +
            units[1] ⊗ ss[2] ⊗ units[3] +
            units[1] ⊗ units[2] ⊗ ss[3]
    ) / 12
    @tensor term_nb1[i' j' k'; i j k] :=
        (nb1x[i' j'; i j] * units[3][k' k] + units[1][i'; i] * nb1y[j' k'; j k]) / 4
    @tensor term_nb2[i' j' k'; i j k] := (nb2[i' k'; i k] * units[2][j'; j]) / 2
    term = term_site + term_nb1 + term_nb2
    return sites => gate_to_mpo(exp(-dt * term))
end
