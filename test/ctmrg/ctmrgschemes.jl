using Test
using Random
using TensorKit
using MPSKit
using PEPSKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg_sequential = CTMRG(; tol=1e-10, verbosity=1, ctmrgscheme=:sequential)
ctm_alg_simultaneous = CTMRG(; tol=1e-10, verbosity=1, ctmrgscheme=:simultaneous)
unitcells = [(1, 1), (3, 4)]

@testset "$(unitcell) unit cell" for unitcell in unitcells
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
    CS_sequential = map(c -> tsvd(c; alg=TensorKit.SVD())[2], env_sequential.corners)
    CS_simultaneous = map(c -> tsvd(c; alg=TensorKit.SVD())[2], env_simultaneous.corners)
    ΔCS = maximum(zip(CS_sequential, CS_simultaneous)) do (c_lm, c_as)
        smallest = infimum(MPSKit._firstspace(c_lm), MPSKit._firstspace(c_as))
        e_old = isometry(MPSKit._firstspace(c_lm), smallest)
        e_new = isometry(MPSKit._firstspace(c_as), smallest)
        return norm(e_new' * c_as * e_new - e_old' * c_lm * e_old)
    end
    @test ΔCS < 1e-2

    TS_sequential = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env_sequential.edges)
    TS_simultaneous = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env_simultaneous.edges)
    ΔTS = maximum(zip(TS_sequential, TS_simultaneous)) do (t_lm, t_as)
        MPSKit._firstspace(t_lm) == MPSKit._firstspace(t_as) || return scalartype(t_lm)(Inf)
        return norm(t_as - t_lm)
    end
    @test ΔTS < 1e-2

    # compare Heisenberg energies
    H = square_lattice_heisenberg(; unitcell)
    E_sequential = costfun(psi, env_sequential, H)
    E_simultaneous = costfun(psi, env_simultaneous, H)
    @test E_sequential ≈ E_simultaneous rtol = 1e-4
end
