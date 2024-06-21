using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using ChainRulesCore
using ChainRulesTestUtils
using KrylovKit
using FiniteDifferences

## Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::CTMRGEnv)
    Ctans = x.corners
    Etans = x.edges
    for i in eachindex(x.corners)
        Ctans[i] = rand_tangent(rng, x.corners[i])
    end
    for i in eachindex(x.edges)
        Etans[i] = rand_tangent(rng, x.edges[i])
    end
    return CTMRGEnv(Ctans, Etans)
end
function ChainRulesTestUtils.test_approx(
    actual::AbstractTensorMap, expected::AbstractTensorMap, msg=""; kwargs...
)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
end
function ChainRulesTestUtils.test_approx(
    actual::InfinitePEPS, expected::InfinitePEPS, msg=""; kwargs...
)
    for i in eachindex(size(actual, 1))
        for j in eachindex(size(actual, 2))
            ChainRulesTestUtils.@test_msg msg isapprox(
                actual[i, j], expected[i, j]; kwargs...
            )
        end
    end
end
function ChainRulesTestUtils.test_approx(
    actual::CTMRGEnv, expected::CTMRGEnv, msg=""; kwargs...
)
    for i in eachindex(actual.corners)
        ChainRulesTestUtils.@test_msg msg isapprox(
            actual.corners[i], expected.corners[i]; kwargs...
        )
    end
    for i in eachindex(actual.edges)
        ChainRulesTestUtils.@test_msg msg isapprox(
            actual.edges[i], expected.edges[i]; kwargs...
        )
    end
end

function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec = mapreduce(vcat, blocks(t)) do (c, b)
        if scalartype(t) <: Real
            return reshape(b, :) .* sqrt(dim(c))
        else
            v = reshape(b, :) .* sqrt(dim(c))
            return vcat(real(v), imag(v))
        end
    end

    function from_vec(x)
        t′ = similar(t)
        T = scalartype(t)
        ctr = 0
        for (c, b) in blocks(t′)
            n = length(b)
            if T <: Real
                copyto!(b, reshape(x[(ctr + 1):(ctr + n)], size(b)) ./ sqrt(dim(c)))
            else
                v = x[(ctr + 1):(ctr + 2n)]
                copyto!(
                    b,
                    complex.(x[(ctr + 1):(ctr + n)], x[(ctr + n + 1):(ctr + 2n)]) ./
                    sqrt(dim(c)),
                )
            end
            ctr += T <: Real ? n : 2n
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

## Model Hamiltonians
# -------------------
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    physical_space = ComplexSpace(2)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return NLocalOperator{NearestNeighbor}(H / 4)
end
function square_lattice_pwave(; t=1, μ=2, Δ=1)
    V = Vect[FermionParity](0 => 1, 1 => 1)
    # on-site
    h0 = TensorMap(zeros, ComplexF64, V ← V)
    block(h0, FermionParity(1)) .= -μ
    H0 = NLocalOperator{OnSite}(h0)
    # two-site (x-direction)
    hx = TensorMap(zeros, ComplexF64, V ⊗ V ← V ⊗ V)
    block(hx, FermionParity(0)) .= [0 -Δ; -Δ 0]
    block(hx, FermionParity(1)) .= [0 -t; -t 0]
    Hx = NLocalOperator{NearestNeighbor}(hx)
    # two-site (y-direction)
    hy = TensorMap(zeros, ComplexF64, V ⊗ V ← V ⊗ V)
    block(hy, FermionParity(0)) .= [0 Δ*im; -Δ*im 0]
    block(hy, FermionParity(1)) .= [0 -t; -t 0]
    Hy = NLocalOperator{NearestNeighbor}(hy)
    return AnisotropicNNOperator(H0, Hx, Hy)
end

## Test models, gradmodes and CTMRG algorithm
# -------------------------------------------
χbond = 2
χenv = 4
Pspaces = [ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0 => χbond / 2, 1 => χbond / 2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)]
models = [square_lattice_heisenberg(), square_lattice_pwave()]
names = ["Heisenberg", "p-wave superconductor"]
Random.seed!(42039482030)
tol = 1e-8
boundary_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=tol, miniter=4, maxiter=100, fixedspace=true, verbosity=0
)
gradmodes = [
    nothing, GeomSum(; tol), ManualIter(; tol), KrylovKit.GMRES(; tol=tol, maxiter=10)
]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD CTMRG energy gradients for $(names[i]) model" for i in eachindex(models)
    Pspace = Pspaces[i]
    Vspace = Pspaces[i]
    Espace = Espaces[i]
    psi_init = InfinitePEPS(Pspace, Vspace, Vspace)
    @testset "$alg_rrule" for alg_rrule in gradmodes
        dir = InfinitePEPS(Pspace, Vspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace, Vspace)
        env = leading_boundary(CTMRGEnv(psi; Venv=Espace), psi, boundary_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.my_retract,
            inner=PEPSKit.my_inner,
        ) do (peps, envs)
            E, g = Zygote.withgradient(peps) do psi
                envs2 = PEPSKit.hook_pullback(
                    leading_boundary, envs, psi, boundary_alg, ; alg_rrule
                )
                return costfun(psi, envs2, models[i])
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
