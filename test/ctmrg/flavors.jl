using Test
using Random
using TensorKit
using MPSKit
using PEPSKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg_sequential = SequentialCTMRG()
ctm_alg_simultaneous = SimultaneousCTMRG()
unitcells = [(1, 1), (3, 4)]
projector_algs = [HalfInfiniteProjector, FullInfiniteProjector]

@testset "$(unitcell) unit cell with $projector_alg" for (unitcell, projector_alg) in
                                                         Iterators.product(
    unitcells, projector_algs
)
    # compute environments
    Random.seed!(32350283290358)
    psi = InfinitePEPS(2, χbond; unitcell)
    env_sequential = leading_boundary(
        CTMRGEnv(psi, ComplexSpace(χenv)), psi, ctm_alg_sequential
    )
    env_simultaneous = leading_boundary(
        CTMRGEnv(psi, ComplexSpace(χenv)), psi, ctm_alg_simultaneous
    )

    # compare norms
    @test abs(norm(psi, env_sequential)) ≈ abs(norm(psi, env_simultaneous)) rtol = 1e-6

    # compare singular values
    CS_sequential = map(c -> tsvd(c)[2], env_sequential.corners)
    CS_simultaneous = map(c -> tsvd(c)[2], env_simultaneous.corners)
    ΔCS = maximum(zip(CS_sequential, CS_simultaneous)) do (c_lm, c_as)
        smallest = infimum(MPSKit._firstspace(c_lm), MPSKit._firstspace(c_as))
        e_old = isometry(MPSKit._firstspace(c_lm), smallest)
        e_new = isometry(MPSKit._firstspace(c_as), smallest)
        return norm(e_new' * c_as * e_new - e_old' * c_lm * e_old)
    end
    @test ΔCS < 1e-2

    TS_sequential = map(t -> tsvd(t)[2], env_sequential.edges)
    TS_simultaneous = map(t -> tsvd(t)[2], env_simultaneous.edges)
    ΔTS = maximum(zip(TS_sequential, TS_simultaneous)) do (t_lm, t_as)
        MPSKit._firstspace(t_lm) == MPSKit._firstspace(t_as) || return scalartype(t_lm)(Inf)
        return norm(t_as - t_lm)
    end
    @test ΔTS < 1e-2

    # compare Heisenberg energies
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    E_sequential = costfun(psi, env_sequential, H)
    E_simultaneous = costfun(psi, env_simultaneous, H)
    @test E_sequential ≈ E_simultaneous rtol = 1e-3
end

# test fixedspace actually fixes space
@testset "Fixedspace truncation using $ctmrg_alg and $projector_alg" for (
    ctmrg_alg, projector_alg
) in Iterators.product(
    [SequentialCTMRG, SimultaneousCTMRG], projector_algs
)
    ctm_alg = ctmrg_alg(;
        tol=1e-6, maxiter=1, verbosity=0, trscheme=FixedSpaceTruncation(), projector_alg
    )
    Ds = fill(2, 3, 3)
    χs = [16 17 18; 15 20 21; 14 19 22]
    psi = InfinitePEPS(Ds, Ds, Ds)
    env = CTMRGEnv(psi, rand(10:20, 3, 3), rand(10:20, 3, 3))
    env2 = leading_boundary(env, psi, ctm_alg)

    # check that the space is fixed
    @test all(space.(env.corners) .== space.(env2.corners))
    @test all(space.(env.edges) .== space.(env2.edges))
end
