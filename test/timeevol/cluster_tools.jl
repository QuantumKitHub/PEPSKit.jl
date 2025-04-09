function inner_prod_cluster(
    Ms1::Vector{T1}, Ms2::Vector{T2}
) where {T1<:AbstractTensorMap,T2<:AbstractTensorMap}
    N = length(Ms1)
    @assert length(Ms2) == N
    @assert all((numout(t) == 1 && numin(t) == 4) for t in Ms1)
    @assert all((numout(t) == 1 && numin(t) == 4) for t in Ms2)
    @assert all(!isdual(space(t, 2)) for t in Ms1)
    @assert all(!isdual(space(t, 2)) for t in Ms2)
    # not the most efficient implementation
    M1, M2 = Ms1[1], deepcopy(Ms2[1])
    for ax in 1:4
        isdual(space(M2, ax)) && twist!(M2, ax)
    end
    @tensor res[-1 -2] := conj(M1[1; 2 3 4 -1]) * M2[1; 2 3 4 -2]
    for i in 2:(N - 1)
        M1, M2 = Ms1[i], deepcopy(Ms2[i])
        for ax in 2:4
            isdual(space(M2, ax)) && twist!(M2, ax)
        end
        @tensor M[-1 -2; -3 -4] := conj(M1[-1; 1 2 3 -3]) * M2[-2; 1 2 3 -4]
        @tensor res[-1 -2] := res[1 2] * M[1 2; -1 -2]
    end
    M1, M2 = Ms1[N], deepcopy(Ms2[N])
    for ax in 2:5
        isdual(space(M2, ax)) && twist!(M2, ax)
    end
    @tensor M[-1 -2] := conj(M1[-1; 1 2 3 4]) * M2[-2; 1 2 3 4]
    return @tensor res[1 2] * M[1 2]
end

function absorb_wts_cluster!(
    Ms::Vector{T}, wts::Vector{DiagonalTensorMap}
) where {T<:AbstractTensorMap}
    revs = [isdual(space(M, 1)) for M in Ms[2:end]]
    for (i, (wt, rev)) in enumerate(zip(wts, revs))
        wtsqrt = sdiag_pow(wt, 0.5)
        if rev
            wtsqrt = permute(wtsqrt, ((2,), (1,)))
        end
        @tensor begin
            Ms[i][-1; -2 -3 -4 -5] = Ms[i][-1; -2 -3 -4 1] * wtsqrt[1; -5]
            Ms[i + 1][-1; -2 -3 -4 -5] = wtsqrt[-1; 1] * Ms[i + 1][1; -2 -3 -4 -5]
        end
    end
end

function fidelity_cluster(
    Ms1::Vector{T1}, Ms2::Vector{T2}
) where {T1<:AbstractTensorMap,T2<:AbstractTensorMap}
    return abs2(inner_prod_cluster(Ms1, Ms2)) /
           (inner_prod_cluster(Ms1, Ms1) * inner_prod_cluster(Ms2, Ms2))
end

function mpo_to_gate3(gs::Vector{T}) where {T<:AbstractTensorMap}
    #= 
    -1         -2        -3
    ↑          ↑          ↑
    g1 ←- 1 ←- g2 ←- 2 ←- g3
    ↑          ↑          ↑
    -4         -5        -6
    =#
    @assert length(gs) == 3
    @tensor gate[-1 -2 -3; -4 -5 -6] := gs[1][-1 -4 1] * gs[2][1 -2 -5 2] * gs[3][2 -3 -6]
    return gate
end
