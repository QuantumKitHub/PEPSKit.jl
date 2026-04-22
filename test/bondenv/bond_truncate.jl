using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using PEPSKit: cost_function_als

Random.seed!(0)
maxiter = 600
check_interval = 20
elt = Float64
d, D = 4, 4
trunc = truncerror(; atol = 1.0e-10) & truncrank(D)
Vphy = Vect[FermionParity](0 => div(d, 2), 1 => div(d, 2))
dD = d * D
Vqro = Vect[FermionParity](0 => div(dD, 2), 1 => div(dD, 2))
d2D = d^2 * D
Vint = Vect[FermionParity](0 => div(d2D, 2), 1 => div(d2D, 2))
for Vl in (Vqro, Vqro'), Vr in (Vqro, Vqro')
    # random positive-definite environment
    Vbond = Vl ⊗ Vr
    Dext = dim(Vbond)
    Vext = Vect[FermionParity](0 => div(Dext, 2) + 1, 1 => div(Dext, 2) + 1)
    @info dim(Vext)
    Z = randn(elt, Vext ← Vbond)
    normalize!(Z, Inf)
    benv = Z' * Z
    # untruncated bond tensors
    a2 = randn(elt, Vl ⊗ Vphy ← Vint)
    b2 = randn(elt, Vint ← Vphy' ⊗ Vr')
    # bond tensor (truncated SVD initialization)
    a2b2 = PEPSKit._combine_ab(a2, b2)
    a0, s, b0 = svd_trunc(permute(a2b2, ((1, 3), (4, 2))); trunc = trunc)
    a0, b0 = PEPSKit.absorb_s(a0, s, b0)
    fid0 = cost_function_als(benv, PEPSKit._combine_ab(a0, b0), a2b2)[2]
    @info "Fidelity of simple SVD truncation = $fid0.\n"
    ss = Dict{String, DiagonalTensorMap}()
    for (label, alg) in (
            ("ALS", ALSTruncation(; trunc, maxiter, check_interval)),
            ("FET", FullEnvTruncation(; trunc, maxiter, check_interval, trunc_init = false)),
        )
        a1, ss[label], b1, info = PEPSKit.bond_truncate(a2, b2, benv, alg)
        @info "$label improved fidelity = $(info.fid)."
        # display(ss[label])
        @test info.fid ≈ cost_function_als(benv, PEPSKit._combine_ab(a1, b1), a2b2)[2]
        @test info.fid > fid0
    end
    @test isapprox(ss["ALS"], ss["FET"], atol = 1.0e-3)
end
