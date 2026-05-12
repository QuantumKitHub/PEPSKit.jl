using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using PEPSKit: cost_function_als, _flip_virtuals!, _cluster_truncate!

Random.seed!(0)
maxiter = 400
check_interval = 20
elt = ComplexF64

#= Index dimensions
                    Dd
                    ↓
                    b
                    ↓ ↘
                    DD′ d
                    ↓
    Dd -←-a-←-DD′-←-M-←- D
          ↘         ↓ ↘
            d       D  d²
Mimicking the situation of an iPEPO with physical dimension d, 
virtual dimension D, updated with an MPO with bond dimension D′.
=#
@testset "3-site iterative optimization ($S)" for S in [Z2Irrep, FermionParity]
    d, D, D′ = 2, 4, 2
    trunc = truncerror(; atol = 1.0e-10) & truncrank(D)
    Dd, DD′ = D * d, D * D′
    hd, hD, hD′ = div(d, 2), div(D, 2), div(D′, 2)
    hDd, hDD = div(Dd, 2), div(DD′, 2)
    VDd = Vect[S](0 => hDd, 1 => hDd)
    VDD = Vect[S](0 => hDD, 1 => hDD)
    VD = Vect[S](0 => hD, 1 => hD)
    Vd = Vect[S](0 => hd, 1 => hd)
    # random positive-definite environment
    Vbond = VDd ⊗ VD' ⊗ VD ⊗ VDd'
    dbond = dim(Vbond)
    Vext = Vect[S](0 => div(dbond, 2) + 2, 1 => div(dbond, 2) + 2)
    Z = randn(elt, Vext ← Vbond)
    benv = Z' * Z
    normalize!(benv, Inf)
    @info "Dimension of benv = $(dbond)"
    # untruncated bond tensor
    Ms = [
        randn(elt, VDd ⊗ Vd ← VDD),
        randn(elt, VDD ⊗ fuse(Vd, Vd) ⊗ VD' ⊗ VD ← VDD'),
        randn(elt, VDD' ⊗ Vd ← VDd),
    ]
    normalize!.(Ms, Inf)
    # Vidal gauge truncation
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    xs = copy.(Ms)
    _flip_virtuals!(xs, flips)
    _cluster_truncate!(xs, fill(trunc, 2))
    _flip_virtuals!(xs, flips)
    cost0, fid0 = cost_function_als(benv, xs, Ms)
    @info "Fidelity of truncated Vidal gauge = $fid0.\n"
    # 3-site iterative optimization
    for alg in (
            ALSProjTruncation(; trunc, maxiter, check_interval),
            ALSTruncation(; trunc, maxiter, check_interval),
        )
        xs, wts, info = PEPSKit.se3site_truncate(Ms, benv, alg)
        @info "Improved fidelity = $(info.fid)."
        @test info.fid ≈ cost_function_als(benv, xs, Ms)[2]
        @test info.fid > fid0
    end
end
