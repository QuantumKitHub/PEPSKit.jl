using Test
using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using Zygote

pspace = ℂ^2
vspace = ℂ^2
χenv = 24

Random.seed!(1564654824)

# Construct random PEPO tensors
O = randn(pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
M = randn(pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')

# Fuse a layer consisting of O-O and MO together
fuser = isometry(vspace ⊗ vspace, fuse(vspace, vspace))
fuser_conj = isometry(vspace' ⊗ vspace', fuse(vspace, vspace)')
@tensor O2[-1 -2; -3 -4 -5 -6] :=
    O[-1 1; 2 4 6 8] *
    O[1 -2; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    fuser_conj[6 7; -5] *
    fuser_conj[8 9; -6]
@tensor MO[-1 -2; -3 -4 -5 -6] :=
    M[-1 1; 2 4 6 8] *
    O[1 -2; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    fuser_conj[6 7; -5] *
    fuser_conj[8 9; -6]

# Create `InfiniteSquareNetwork`s of of both options
network = InfiniteSquareNetwork(InfinitePEPO(fill(O, 1, 1, 2)));
network_fused = InfiniteSquareNetwork(InfinitePEPO(O2));

# cover all different flavors
ctm_styles = [:sequential, :simultaneous]
projector_algs = [:halfinfinite, :fullinfinite]

@testset "PEPO layers CTMRG contraction using $alg with $projector_alg" for (
    alg, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    env, = leading_boundary(
        CTMRGEnv(network, χenv), network; alg, maxiter=150, projector_alg
    )
    env_fused, = leading_boundary(
        CTMRGEnv(network_fused, χenv), network_fused; alg, maxiter=150, projector_alg
    )

    m = PEPSKit.contract_local_tensor((1, 1, 1), M, network, env)
    m_fused = PEPSKit.contract_local_tensor((1, 1, 1), MO, network_fused, env_fused)

    nrm = PEPSKit._contract_site((1, 1), network, env)
    nrm_fused = PEPSKit._contract_site((1, 1), network_fused, env_fused)

    @test (m / nrm) ≈ (m_fused / nrm_fused) atol = 1e-9
end

ψ = ones(pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
@tensor Oψ[-1; -3 -4 -5 -6] :=
    O[-1 1; 2 4 6 8] *
    ψ[1; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    fuser_conj[6 7; -5] *
    fuser_conj[8 9; -6]
@tensor Mψ[-1; -3 -4 -5 -6] :=
    M[-1 1; 2 4 6 8] *
    ψ[1; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    fuser_conj[6 7; -5] *
    fuser_conj[8 9; -6]

O_stack = fill(O, 1, 1, 2)
O_stack[1, 1, 2] = dagger(O)
OOdag = InfinitePEPO(O_stack)

network_O = InfiniteSquareNetwork(InfinitePEPS(ψ), OOdag, InfinitePEPS(ψ))
network_fused_MO = InfiniteSquareNetwork(InfinitePEPS(Mψ), InfinitePEPS(Oψ))
network_fused_OO = InfinitePEPS(Oψ)

@testset "PEPO layers CTMRG contraction with dagger using $alg with $projector_alg" for (
    alg, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    env, = leading_boundary(
        CTMRGEnv(network_O, χenv), network_O; alg, maxiter=250, projector_alg
    )
    env_fused, = leading_boundary(
        CTMRGEnv(network_fused_OO, χenv), network_fused_OO; alg, maxiter=250, projector_alg
    )

    m = PEPSKit.contract_local_tensor((1, 1, 1), M, network_O, env)
    m_fused = network_value(network_fused_MO, env_fused)

    nrm = PEPSKit._contract_site((1, 1), network_O, env)
    nrm_fused = network_value(network_fused_OO, env_fused)

    @test (m / nrm) ≈ (m_fused / nrm_fused) atol = 1e-7
end

@testset "mpotensor for PEPOLayersSandwich" begin
    network = InfiniteSquareNetwork(OOdag)

    # Fuse the two physical legs of the PEPO to convert it to a PEPS
    F = isomorphism(fuse(codomain(O)), codomain(O))
    @tensor O_fused[-1; -2 -3 -4 -5] := O[1 2; -2 -3 -4 -5] * F[-1; 1 2]
    network_fused = InfiniteSquareNetwork(InfinitePEPS(O_fused))

    # Construct the mpotensor of the double-layer and compare
    mpo = InfiniteSquareNetwork(map(PEPSKit.mpotensor, PEPSKit.unitcell(network)))
    mpo_fused = InfiniteSquareNetwork(
        map(PEPSKit.mpotensor, PEPSKit.unitcell(network_fused))
    )
    @test mpo ≈ mpo_fused
end
