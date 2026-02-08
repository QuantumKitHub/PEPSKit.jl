# North edge
# ----------

"""
    renormalize_north_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_north_edge(E_north, P_left, P_right, A::P)

Absorb a local effective tensor `A` into the north edge using the given projectors and
environment tensors.

```
          |~~~~~~~| -- E_north -- |~~~~~~|
    out-- |P_right|       |       |P_left| --in
          |~~~~~~~| --    A    -- |~~~~~~|
                          |
```
"""
function renormalize_north_edge(
        (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork
    )
    return renormalize_north_edge(
        env.edges[NORTH, _prev(row, end), col],
        P_left[NORTH, row, col],
        P_right[NORTH, row, _prev(col, end)],
        network[row, col], # so here it's fine
    )
end
function renormalize_north_edge(E_north, P_left, P_right, A)
    A_west = _rotl90_localsandwich(A)
    return renormalize_west_edge(E_north, P_left, P_right, A_west)
end
# specialize PartitionFunction to avoid permute(A)
function renormalize_north_edge(E_north::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @tensor begin
        temp = permute(E_north, ((2, 1), (3,))) # impose D_N as 1st leg
        PE[D_N D_E; χNW χ_E] := temp[D_N χNW; χNE] * P_left[χNE D_E; χ_E]
        PEA[D_W χNW; D_S χ_E] := A[D_W D_S; D_N D_E] * PE[D_N D_E; χNW χ_E]
        P_rightp = permute(P_right, ((1,), (3, 2)))
        edge[χ_W D_S; χ_E] := P_rightp[χ_W; D_W χNW] * PEA[D_W χNW; D_S χ_E]
    end
end

@generated function renormalize_north_edge(
        E_north::CTMRGEdgeTensor{T, S, N}, P_left, P_right, A::PEPOSandwich{H}
    ) where {T, S, N, H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :S, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :W, :W, H)
    E_north_e = _pepo_edge_expr(:E_north, :W, :E, :N, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :E, :E, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_north_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

# East edge
# ---------

"""
    renormalize_east_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_east_edge(E_east, P_left, P_right, A::P)

Absorb a local effective tensor into the east edge using the given projectors and
environment tensors.

```
          out
           |
      [~P_right~]
       |       |
    -- A -- E_east
       |       |
      [~~P_left~]
           |
           in
```
"""
function renormalize_east_edge(
        (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork
    )
    return renormalize_east_edge(
        env.edges[EAST, row, _next(col, end)],
        P_left[EAST, row, col],
        P_right[EAST, _prev(row, end), col],
        network[row, col],
    )
end
function renormalize_east_edge(E_east, P_left, P_right, A)
    A_west = _rot180_localsandwich(A)
    return renormalize_west_edge(E_east, P_left, P_right, A_west)
end
# specialize PartitionFunction to avoid permute(A)
function renormalize_east_edge(E_east::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @tensor begin
        temp = permute(P_right, ((3, 1), (2,)))  # impose D_N as 1st leg
        PE[D_N D_E; χN χSE] := temp[D_N χN; χNE] * E_east[χNE D_E; χSE]
        PEA[D_W χN; χSE D_S] := A[D_W D_S; D_N D_E] * PE[D_N D_E; χN χSE]
        edge[χ_N D_W; χ_S] := PEA[D_W χ_N; χSE D_S] * P_left[χSE D_S; χ_S]
    end
end

@generated function renormalize_east_edge(
        E_east::CTMRGEdgeTensor{T, S, N}, P_left, P_right, A::PEPOSandwich{H}
    ) where {T, S, N, H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :W, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :N, :N, H)
    E_east_e = _pepo_edge_expr(:E_east, :N, :S, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :S, :S, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_east_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

# South edge
# ----------

"""
    renormalize_south_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_south_edge(E_south, P_left, P_right, A::P)

Absorb a local effective tensor into the south edge using the given projectors and
environment tensors.

```
                        |
         |~~~~~~| --    A    -- |~~~~~~~|
    in-- |P_left|       |       |P_right| --out
         |~~~~~~| -- E_south -- |~~~~~~~|
```
"""
function renormalize_south_edge(
        (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork
    )
    return renormalize_south_edge(
        env.edges[SOUTH, _next(row, end), col],
        P_left[SOUTH, row, col],
        P_right[SOUTH, row, _next(col, end)],
        network[row, col],
    )
end
function renormalize_south_edge(E_south, P_left, P_right, A)
    A_west = _rotr90_localsandwich(A)
    return renormalize_west_edge(E_south, P_left, P_right, A_west)
end
# specialize PartitionFunction to avoid permute(A)
function renormalize_south_edge(E_south::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @tensor begin
        P_leftp = permute(P_left, ((3, 2), (1,)))  # impose χ_W as 1st leg
        PE[χ_W χSE; D_W D_S] := P_leftp[χ_W D_W; χSW] * E_south[χSE D_S; χSW]
        PEA[χ_W D_N; χSE D_E] := PE[χ_W χSE; D_W D_S] * A[D_W D_S; D_N D_E]
        edge[χ_E D_N; χ_W] := PEA[χ_W D_N; χSE D_E] * P_right[χ_E; χSE D_E]
    end
end

@generated function renormalize_south_edge(
        E_south::CTMRGEdgeTensor{T, S, N}, P_left, P_right, A::PEPOSandwich{H}
    ) where {T, S, N, H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :N, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :E, :E, H)
    E_south_e = _pepo_edge_expr(:E_south, :E, :W, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :W, :W, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_south_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

# West edge
# ---------

"""
    renormalize_west_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_west_edge(E_west, P_left, P_right, A::P)

Absorb a local effective tensor into the west edge using the given projectors and
environment tensors.

```
          in
          |
     [~~P_left~]
      |       |
    E_west -- A --
      |       |
     [~P_right~]
          |
         out
```
"""
function renormalize_west_edge(  # For simultaneous CTMRG scheme
        (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork,
    )
    return renormalize_west_edge(
        env.edges[WEST, row, _prev(col, end)],
        P_left[WEST, row, col],
        P_right[WEST, _next(row, end), col],
        network[row, col],
    )
end
function renormalize_west_edge(  # For sequential CTMRG scheme
        (row, col), env::CTMRGEnv, projectors, network::InfiniteSquareNetwork,
    )
    return renormalize_west_edge(
        env.edges[WEST, row, _prev(col, end)],
        projectors[1][row],
        projectors[2][_next(row, end)],
        network[row, col],
    )
end
function renormalize_west_edge(
        E_west::CTMRG_PEPS_EdgeTensor, P_left, P_right, A::PEPSSandwich
    )
    # starting with P_bottom to save one permute in the end
    return @tensor begin
        # already putting χE in front here to make next permute cheaper
        PE[χS χNW DSb DWb; DSt DWt] := P_right[χS; χSW DSt DSb] * E_west[χSW DWt DWb; χNW]
        PEket[χS χNW DNt DEt; DSb DWb d] :=
            PE[χS χNW DSb DWb; DSt DWt] * ket(A)[d; DNt DEt DSt DWt]
        corner[χS DEt DEb; χNW DNt DNb] :=
            PEket[χS χNW DNt DEt; DSb DWb d] * conj(bra(A)[d; DNb DEb DSb DWb])
        edge[χS DEt DEb; χN] := corner[χS DEt DEb; χNW DNt DNb] * P_left[χNW DNt DNb; χN]
    end
end
function renormalize_west_edge(E_west::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @tensor begin
        PE[χ_S χNW; D_W D_S] := P_right[χ_S; χSW D_S] * E_west[χSW D_W; χNW]
        PEA[χ_S D_E; χNW D_N] := PE[χ_S χNW; D_W D_S] * A[D_W D_S; D_N D_E]
        edge[χ_S D_E; χ_N] := PEA[χ_S D_E; χNW D_N] * P_left[χNW D_N; χ_N]
    end
end

@generated function renormalize_west_edge(
        E_west::CTMRGEdgeTensor{T, S, N}, P_left, P_right, A::PEPOSandwich{H}
    ) where {T, S, N, H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :E, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :S, :S, H)
    E_west_e = _pepo_edge_expr(:E_west, :S, :N, :W, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :N, :N, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_west_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end
