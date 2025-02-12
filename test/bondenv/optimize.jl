using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit

function _postprocess(
    a::AbstractTensorMap{T,S,2,1},
    s::AbstractTensorMap{T,S,1,1},
    b::AbstractTensorMap{T,S,1,2},
) where {T<:Number,S<:ElementarySpace}
    a, b = PEPSKit.absorb_s(a, s, b)
    return permute(a, (1, 2, 3)), permute(b, (1, 2, 3))
end

Random.seed!(10)
trscheme = truncerr(1e-10) & truncdim(8)
Vext = ℂ[FermionParity](0 => 100, 1 => 100)
Vint = ℂ[FermionParity](0 => 6, 1 => 6)
Vphy = ℂ[FermionParity](0 => 1, 1 => 2)
Vbond = Vint' ⊗ Vint

# random positive-definite environment
Z = randn(Float64, Vext ← Vbond)
env = Z' * Z
@assert env ≈ env'
D, U = eigh(env)
@assert all(D.data .>= 0.0)

# untruncated bond tensor
a2b2 = randn(Float64, Vint' ⊗ Vphy ⊗ Vphy ⊗ Vint)
a2, s, b2 = tsvd(a2b2, ((1, 2), (3, 4)))
a2, b2 = _postprocess(a2, s, b2)
# bond tensor (truncated SVD initialization)
a0, s, b0 = tsvd(a2b2, ((1, 2), (3, 4)); trunc=trscheme)
a0, b0 = _postprocess(a0, s, b0)
fid0 = PEPSKit.fidelity(env, PEPSKit._combine_ab(a0, b0), a2b2)
@info "Fidelity of simple SVD truncation = $fid0.\n"

maxiter = 200
ss = Dict{String,DiagonalTensorMap}()
for (label, alg) in (
    ("ALS", ALSTruncation(; trscheme, maxiter, check_int=10)),
    ("FET", FullEnvTruncation(; trscheme, maxiter, check_int=10)),
)
    a1, ss[label], b1, info = PEPSKit.bond_optimize(env, a2, b2, alg)
    @info "$label improved fidelity = $(info.fid)."
    display(ss[label])
    a1, b1 = _postprocess(a1, ss[label], b1)
    @test info.fid ≈ PEPSKit.fidelity(env, PEPSKit._combine_ab(a1, b1), a2b2)
    @test info.fid > fid0
end
# @test isapprox(ss["ALS"], ss["FET"], atol=1e-3)
