"""
Bundle the zip-up contraction and optional DMRG refinement
algorithms used after each finite window MPO-MPS contraction.
"""
struct WindowApprox{Z, D}
    zipup::Z
    dmrg::D
end

# TODO: generalize the following to multi-layer networks

"""
Apply a finite MPO to a finite MPS with zip-up truncation and optional DMRG refinement.
"""
function _approximate(W::FiniteMPO, ψ::FiniteMPS, alg::WindowApprox)
    ψ′, = approximate((W, ψ), alg.zipup)
    isnothing(alg.dmrg) && return ψ′
    ψ′, = approximate(ψ′, (W, ψ), alg.dmrg)
    return ψ′
end

"""
Build the finite MPS representing the north CTMRG boundary of a window.

Convention of CTM tensors on the north boundary is
```
    [1; 2]      [1 2; 3]        [1; 2]
    C₁-←-2      1-←-E₁-←-3      1-←-C₂
    ↓               ↓               ↑
    1               2               2
```
Leg 2 of C₂ needs to be flipped to have a non-dual physical space.
"""
function _north_boundary_mps(
        env::CTMRGEnv, row::Int, colrange::UnitRange{Int},
    )
    r = row - 1
    cmin, cmax = first(colrange), last(colrange)
    Cwest = insertleftunit(corner(env, NORTHWEST, r, cmin - 1), 1)
    tensors = [Cwest]
    append!(tensors, (edge(env, NORTH, r, col) for col in colrange))
    Ceast = repartition(
        flip(corner(env, NORTHEAST, r, cmax + 1), 2), 2, 0
    )
    push!(tensors, insertleftunit(Ceast, 3))
    return FiniteMPS(tensors)
end

"""
Build the finite MPS representing the south CTMRG boundary of a window,
but with dual physical legs, and sites ordered from east to west.

Convention of CTM tensors on the south boundary is
```
    [1; 2]      [1 2; 3]        [1; 2]
    2               2               1
    ↓               ↓               ↑
    C₄-→-1      3-→-E₃-→-1      2-→-C₃
```
Leg 1 of C₃ needs to be flipped to have a dual physical space.

North site `k` pairs with south site `N + 1 - k`.
```
    west                 east
    north:  1 ← … ← N - 1 ← N
    south:  N → … → 2     → 1
```
Viewed after a 180° rotation of the entire network, this is a north boundary ordered from west to east as usual, only with dual physical legs.
Because of this reversed site order, `AL` tensors lie to the right (east) of the canonical center in the window, while `AR` tensors lie to its left (west).
"""
function _south_boundary_mps(env::CTMRGEnv, row::Int, colrange::UnitRange{Int})
    r = row + 1
    cmin, cmax = first(colrange), last(colrange)
    Ceast = insertleftunit(flip(corner(env, SOUTHEAST, r, cmax + 1), 1), 1)
    tensors = [Ceast]
    append!(tensors, (edge(env, SOUTH, r, col) for col in reverse(colrange)))
    Cwest = repartition(corner(env, SOUTHWEST, r, cmin - 1), 2, 0)
    push!(tensors, insertleftunit(Cwest, 3))
    return FiniteMPS(tensors)
end
