using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit

Random.seed!(10)
maxiter = 500
check_interval = 20
trscheme = truncerr(1e-10) & truncdim(8)
Vext = Vect[FermionParity](0 => 100, 1 => 100)
Vint = Vect[FermionParity](0 => 6, 1 => 6)
Vphy = Vect[FermionParity](0 => 1, 1 => 2)
for Vbondl in (Vint, Vint'), Vbondr in (Vint, Vint')
    Vbond = Vbondl ⊗ Vbondr
    # random positive-definite environment
    Z = randn(Float64, Vext ← Vbond)
    benv = Z' * Z
    # untruncated bond tensor
    a2b2 = randn(Float64, Vbondl ⊗ Vphy ← Vphy' ⊗ Vbondr')
    a2, s, b2 = tsvd(a2b2)
    a2, b2 = PEPSKit.absorb_s(a2, s, b2)
    # bond tensor (truncated SVD initialization)
    a0, s, b0 = tsvd(a2b2; trunc=trscheme)
    a0, b0 = PEPSKit.absorb_s(a0, s, b0)
    fid0 = PEPSKit.fidelity(benv, PEPSKit._combine_ab(a0, b0), a2b2)
    @info "Fidelity of simple SVD truncation = $fid0.\n"
    ss = Dict{String,DiagonalTensorMap}()
    for (label, alg) in (
        ("ALS", ALSTruncation(; trscheme, maxiter, check_interval)),
        ("FET", FullEnvTruncation(; trscheme, maxiter, check_interval)),
    )
        a1, ss[label], b1, info = PEPSKit.bond_truncate(a2, b2, benv, alg)
        @info "$label improved fidelity = $(info.fid)."
        display(ss[label])
        a1, b1 = PEPSKit.absorb_s(a1, ss[label], b1)
        @test info.fid ≈ PEPSKit.fidelity(benv, PEPSKit._combine_ab(a1, b1), a2b2)
        @test info.fid > fid0
    end
    @test isapprox(ss["ALS"], ss["FET"], atol=1e-3)
end
