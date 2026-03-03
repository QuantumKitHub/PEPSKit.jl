"""
$(TYPEDEF)

Abstract super type for the collection of
Trotter evolution MPOs acting on 3 or more sites.
"""
abstract type TrotterMPOs end

Base.getindex(gate::TrotterMPOs, args...) = Base.getindex(gate.mpos, args...)

"""
    struct TrotterMPOs2ndNeighbor{T}

Collection of all Trotter evolution MPOs obtained from a Hamiltonian
containing up to 2nd nearest neighbor terms.

Before exponentiating, terms in the Hamiltonian are organized as
```
    H = ∑ᵢⱼ(┘ᵢⱼ + ┐ᵢⱼ + ┌ᵢⱼ + └ᵢⱼ)
```
where `┘`, `┐`, `┌`, `└` refer to the following 3-site clusters 
```
        3   3---2   2---1   1
        |       |   |       |
    1---2       1   3       2---3
```
Then each Trotter MPO is `exp(-dt * ┘ᵢⱼ)`, etc.
"""
struct TrotterMPOs2ndNeighbor{T} <: TrotterMPOs
    mpos::T
end

function TrotterMPOs2ndNeighbor(H::LocalOperator, dt::Number)
    mpos = [
        _get_gatempos_se(H, dt),
        _get_gatempos_se(rotl90(H), dt),
        _get_gatempos_se(rot180(H), dt),
        _get_gatempos_se(rotr90(H), dt)
    ]
    return TrotterMPOs2ndNeighbor(mpos)
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
    term = _get_se3site_term(ham, row, col)
    return gate_to_mpo3(exp(-dt * term))
end

"""
Construct the 3-site gate MPOs on the southeast cluster 
for 3-site simple update on square lattice.
"""
function _get_gatempos_se(ham::LocalOperator, dt::Number)
    Nr, Nc = size(ham.lattice)
    return collect(_get_gatempo_se(ham, dt, r, c) for r in 1:Nr, c in 1:Nc)
end
