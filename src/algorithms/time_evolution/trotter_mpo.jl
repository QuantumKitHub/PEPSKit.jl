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
