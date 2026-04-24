using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using PEPSKit: bond_truncate, cost_function_als
using PEPSKit: _combine_ket, _combine_ket_for_svd

Random.seed!(0)
maxiter = 600
check_interval = 30
elt = Float64
# simulating the situation of applying a 2-site gate
# to a bond with virtual dimension D, physical dimension d.
d, D = 2, 4
trunc = truncerror(; atol = 1.0e-10) & truncrank(D)
Vphy = Vect[FermionParity](0 => div(d, 2), 1 => div(d, 2))
Vqro = Vect[FermionParity](0 => div(d * D, 2), 1 => div(d * D, 2))
# virtual dimension of gate MPO is d^2
Vint = Vect[FermionParity](0 => div(d^2 * D, 2), 1 => div(d^2 * D, 2))
for Vl in (Vqro, Vqro'), Vr in (Vqro, Vqro')
    # random positive-definite environment
    Vbond = Vl ⊗ Vr
    Dext = dim(Vbond)
    Vext = Vect[FermionParity](0 => div(Dext, 2) + 1, 1 => div(Dext, 2) + 1)
    Z = randn(elt, Vext ← Vbond)
    normalize!(Z, Inf)
    benv = Z' * Z
    @info "Dimension of benv = $(Dext)"
    # untruncated bond tensors
    a2 = randn(elt, Vl ⊗ Vphy ← Vint)
    b2 = randn(elt, Vint ← Vphy' ⊗ Vr')
    # bond tensor (truncated SVD initialization)
    a2b2 = _combine_ket(a2, b2)
    a0, s, b0 = svd_trunc(permute(a2b2, ((1, 3), (4, 2))); trunc = trunc)
    a0, b0 = PEPSKit.absorb_s(a0, s, b0)
    fid0 = cost_function_als(benv, _combine_ket(a0, b0), a2b2)[2]
    @info "Fidelity of simple SVD truncation = $fid0.\n"
    ss = Dict{String, DiagonalTensorMap}()
    # FET is slower when d is large
    for (label, alg) in (
            ("ALS", ALSTruncation(; trunc, maxiter, check_interval)),
            ("FET", FullEnvTruncation(; trunc, maxiter, check_interval, trunc_init = false)),
        )
        a1, ss[label], b1, info = bond_truncate(a2, b2, benv, alg)
        @info "$label improved fidelity = $(info.fid)."
        # display(ss[label])
        @test info.fid ≈ cost_function_als(benv, _combine_ket(a1, b1), a2b2)[2]
        @test info.fid > fid0
    end
    @test isapprox(ss["ALS"], ss["FET"], atol = 1.0e-3)
end
