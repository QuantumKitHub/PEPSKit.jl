using Test
using Random
using LinearAlgebra
using TensorKit
using KrylovKit
using OptimKit
using Zygote
using PEPSKit
import PEPSKit: unitcell

χenv = 24
T = ComplexF64

I = fℤ₂
pspace = Vect[I](0 => 1, 1 => 1)
vspace = Vect[I](0 => 1, 1 => 1)
envspace = Vect[I](0 => χenv, 1 => χenv)

Random.seed!(97646475)

# Construct random PEPO tensors
O = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
O = O + twist(flip(PEPSKit._dag(O), 3:6), [4 6])
M = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
M = M + twist(flip(PEPSKit._dag(M), 3:6), [4 6])

# Fuse a layer consisting of O-O and MO together
fuser = isomorphism(T, vspace ⊗ vspace, fuse(vspace, vspace))
fuser_adj = isomorphism(T, vspace' ⊗ vspace, fuse(vspace', vspace))
@tensor O2[-1 -2; -3 -4 -5 -6] :=
    O[-1 1; 2 4 6 8] *
    O[1 -2; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    conj(fuser[6 7; -5]) *
    conj(fuser[8 9; -6])
@tensor MO[-1 -2; -3 -4 -5 -6] :=
    M[-1 1; 2 4 6 8] *
    O[1 -2; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    conj(fuser[6 7; -5]) *
    conj(fuser[8 9; -6])

# Create `InfiniteSquareNetwork`s of of both options
O_stack = fill(O, 1, 1, 2)
network = InfiniteSquareNetwork(InfinitePEPO(O_stack));
network_fused = InfiniteSquareNetwork(InfinitePEPO(O2));

ψ = randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
ψ = ψ / norm(ψ)
@tensor Oψ[-1; -3 -4 -5 -6] :=
    O[-1 1; 2 4 6 8] *
    ψ[1; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    conj(fuser[6 7; -5]) *
    conj(fuser[8 9; -6])
@tensor Mψ[-1; -3 -4 -5 -6] :=
    M[-1 1; 2 4 6 8] *
    ψ[1; 3 5 7 9] *
    fuser[2 3; -3] *
    fuser[4 5; -4] *
    conj(fuser[6 7; -5]) *
    conj(fuser[8 9; -6])
ϕ = randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
ϕ = ϕ / norm(ϕ)
@tensor Odagϕ[-1; -3 -4 -5 -6] :=
    twist(PEPSKit._dag(O), 1:4)[-1 1; 2 4 6 8] *
    ϕ[1; 3 5 7 9] *
    fuser_adj[2 3; -3] *
    fuser_adj[4 5; -4] *
    conj(fuser_adj[6 7; -5]) *
    conj(fuser_adj[8 9; -6])

(Nr, Nc) = (1, 1)
Oinf = InfinitePEPO(O; unitcell=(Nr, Nc, 1))
O_stack = fill(O, Nr, Nc, 2)
O_stack[:, :, 2] .= unitcell(adjoint(Oinf))
OOdag = InfinitePEPO(O_stack)

network_O = InfiniteSquareNetwork(InfinitePEPS(ψ), OOdag, InfinitePEPS(ψ))
network_fused_MO = InfiniteSquareNetwork(InfinitePEPS(Mψ), InfinitePEPS(Oψ))
network_fused_OO = InfinitePEPS(Oψ)

# cover all different flavors
ctm_styles = [:sequential, :simultaneous]
projector_algs = [:halfinfinite, :fullinfinite]

@testset "mpotensor for PEPOTraceSandwich using $alg with $projector_alg" for (
    alg, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    # Test whether the mpotensor of a double layer of PEPOs is the same as in the case where the layers are fused
    mpo = InfiniteSquareNetwork(map(PEPSKit.mpotensor, PEPSKit.unitcell(network)))
    mpo_fused = InfiniteSquareNetwork(
        map(PEPSKit.mpotensor, PEPSKit.unitcell(network_fused))
    )
    env, = leading_boundary(CTMRGEnv(mpo, envspace), mpo; alg, maxiter=250, projector_alg)
    env_fused, = leading_boundary(
        CTMRGEnv(mpo_fused, envspace), mpo_fused; alg, maxiter=250, projector_alg
    )

    @test network_value(mpo, env) ≈ network_value(mpo_fused, env_fused)
end

@testset "PEPO layers CTMRG contraction using $alg with $projector_alg" for (
    alg, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    # Test whether calculating the environment of a double layer of PEPOs results in the same expectation value as in the case where the layers are fused
    env, = leading_boundary(
        CTMRGEnv(network, envspace), network; alg, maxiter=250, projector_alg
    )
    env_fused, = leading_boundary(
        CTMRGEnv(network_fused, envspace), network_fused; alg, maxiter=250, projector_alg
    )

    m = PEPSKit.contract_local_tensor((1, 1, 1), M, network, env)
    m_fused = PEPSKit.contract_local_tensor((1, 1, 1), MO, network_fused, env_fused)

    nrm = PEPSKit._contract_site((1, 1), network, env)
    nrm_fused = PEPSKit._contract_site((1, 1), network_fused, env_fused)
    @test (m / nrm) ≈ (m_fused / nrm_fused) atol = 1e-9
end

projector_alg = projector_algs[1] # only use :halfinfinite for this test due to convergence issues
@testset "Test adjoint of an InfinitePEPO using $alg with $projector_alg" for alg in
                                                                              ctm_styles
    # Test the definition of the adjoint of an operator, i.e. dot(psi, O, phi) == dot(psi, O * phi) == dot(O' * psi, phi) for any two states psi and phi.
    network_ψOϕ = InfiniteSquareNetwork(
        InfinitePEPS(ψ), InfinitePEPO(PEPSKit._dag(O)), InfinitePEPS(ϕ)
    )
    network_ψ_Odagϕ = InfiniteSquareNetwork(InfinitePEPS(ψ), InfinitePEPS(Odagϕ))
    network_Oψ_ϕ = InfiniteSquareNetwork(InfinitePEPS(Oψ), InfinitePEPS(ϕ))

    env_ψOϕ, = leading_boundary(
        CTMRGEnv(network_ψOϕ, envspace),
        network_ψOϕ;
        alg,
        maxiter=500,
        projector_alg,
        verbosity=2,
    )
    env_ψ_Odagϕ, = leading_boundary(
        CTMRGEnv(network_ψ_Odagϕ, envspace),
        network_ψ_Odagϕ;
        alg,
        maxiter=500,
        projector_alg,
        verbosity=2,
    )
    env_Oψ_ϕ, = leading_boundary(
        CTMRGEnv(network_Oψ_ϕ, envspace),
        network_Oψ_ϕ;
        alg,
        maxiter=500,
        projector_alg,
        verbosity=2,
    )

    overlap1 = network_value(network_ψOϕ, env_ψOϕ)
    overlap2 = network_value(network_ψ_Odagϕ, env_ψ_Odagϕ)
    overlap3 = network_value(network_Oψ_ϕ, env_Oψ_ϕ)

    @test overlap1 ≈ overlap2 atol = 5e-3
    @test overlap1 ≈ overlap3 atol = 5e-3
end

@testset "PEPO layers CTMRG contraction with its adjoint using $alg with $projector_alg" for (
    alg, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    # Test whether calculating the environment of `PEPOSandwich` results in the same expectation value as when the PEPO is fused with the PEPS
    env, = leading_boundary(
        CTMRGEnv(network_O, envspace), network_O; alg, maxiter=400, projector_alg
    )
    env_fused, = leading_boundary(
        CTMRGEnv(network_fused_OO, envspace),
        network_fused_OO;
        alg,
        maxiter=400,
        projector_alg,
    )

    m = PEPSKit.contract_local_tensor((1, 1, 1), M, network_O, env)
    m_fused = network_value(network_fused_MO, env_fused)

    nrm = PEPSKit._contract_site((1, 1), network_O, env)
    nrm_fused = network_value(network_fused_OO, env_fused)

    @test (m / nrm) ≈ (m_fused / nrm_fused) atol = 5e-5
end
