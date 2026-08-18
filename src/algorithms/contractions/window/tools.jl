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
function _approximate_window_step(W::FiniteMPO, ψ::FiniteMPS, alg::WindowApprox)
    ψ′, = approximate((W, ψ), alg.zipup)
    isnothing(alg.dmrg) && return ψ′
    ψ′, = approximate(ψ′, (W, ψ), alg.dmrg)
    return ψ′
end

"""
Convert a south-boundary MPS tensor into the stored bra representation expected by `dot` using a planar repartition.
"""
function _bra_mps_tensor(A::MPSTensor)
    return repartition(A', 2, 1; copy = true)
end

"""
Construct the planar adjoint of a finite MPO while restoring MPSKit's local MPO leg partition.
"""
function _adjoint_mpo(W::FiniteMPO)
    return FiniteMPO(map(A -> transpose(A', ((3, 1), (4, 2)); copy = true), parent(W)))
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
"""
function _north_boundary_mps(
        env::CTMRGEnv, row::Int, colrange::UnitRange{Int},
    )
    r = row - 1
    cmin, cmax = first(colrange), last(colrange)
    Cwest = insertleftunit(corner(env, NORTHWEST, r, cmin - 1), 1)
    tensors = [Cwest]
    append!(tensors, (edge(env, NORTH, r, col) for col in colrange))
    Ceast = repartition(corner(env, NORTHEAST, r, cmax + 1), 2, 0)
    push!(tensors, insertleftunit(Ceast, 3))
    return FiniteMPS(tensors)
end

"""
Build the finite MPS representing the adjointed south CTMRG boundary of a window.

Convention of CTM tensors on the south boundary is
```
    [1; 2]      [1 2; 3]        [1; 2]
    2               2               1
    ↓               ↓               ↑
    C₄-→-1      3-→-E₃-→-1      2-→-C₃
```
Their adjoints are
```
    [1; 2]      [1; 2 3]        [1; 2]
    C̄₄-←-2      1-←-Ē₃-←-2      1-←-C̄₃
    ↓               ↓               ↑
    1               3               2
```
The edge tensors then need a further repartition of indices.
"""
function _south_boundary_mps(
        env::CTMRGEnv, row::Int, colrange::UnitRange{Int},
    )
    r = row + 1
    cmin, cmax = first(colrange), last(colrange)
    Cwest = insertleftunit(corner(env, SOUTHWEST, r, cmin - 1)', 1)
    tensors = [Cwest]
    append!(tensors, (_bra_mps_tensor(edge(env, SOUTH, r, col)) for col in colrange))
    Ceast = repartition(corner(env, SOUTHEAST, r, cmax + 1)', 2, 0; copy = true)
    # The planar bend of the adjointed southeast corner carries a twist
    push!(tensors, insertleftunit(twist!(Ceast, 1), 3))
    return FiniteMPS(tensors)
end
