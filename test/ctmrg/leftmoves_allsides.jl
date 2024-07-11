using Test
using Random
using TensorKit
using MPSKit
using PEPSKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg_leftmoves = CTMRG(; tol=1e-10, verbosity=1, ctmrgscheme=:LeftMoves)
ctm_alg_allsides = CTMRG(; tol=1e-10, verbosity=1, ctmrgscheme=:AllSides)
unitcells = [(1, 1), (3, 4)]

@testset "$(unitcell) unit cell" for unitcell in unitcells
    # compute environments
    Random.seed!(32350283290358)
    psi = InfinitePEPS(2, χbond; unitcell)
    env_leftmoves = leading_boundary(
        CTMRGEnv(psi; Venv=ComplexSpace(χenv)), psi, ctm_alg_leftmoves
    )
    env_allsides = leading_boundary(
        CTMRGEnv(psi; Venv=ComplexSpace(χenv)), psi, ctm_alg_allsides
    )

    # compare norms
    @test abs(norm(psi, env_leftmoves)) ≈ abs(norm(psi, env_allsides)) rtol = 1e-6

    # compare singular values
    CS_leftmoves = map(c -> tsvd(c; alg=TensorKit.SVD())[2], env_leftmoves.corners)
    CS_allsides = map(c -> tsvd(c; alg=TensorKit.SVD())[2], env_allsides.corners)
    ΔCS = maximum(zip(CS_leftmoves, CS_allsides)) do (c_lm, c_as)
        smallest = infimum(MPSKit._firstspace(c_lm), MPSKit._firstspace(c_as))
        e_old = isometry(MPSKit._firstspace(c_lm), smallest)
        e_new = isometry(MPSKit._firstspace(c_as), smallest)
        return norm(e_new' * c_as * e_new - e_old' * c_lm * e_old)
    end
    @test ΔCS < 1e-2

    TS_leftmoves = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env_leftmoves.edges)
    TS_allsides = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env_allsides.edges)
    ΔTS = maximum(zip(TS_leftmoves, TS_allsides)) do (t_lm, t_as)
        MPSKit._firstspace(t_lm) == MPSKit._firstspace(t_as) || return scalartype(t_lm)(Inf)
        return norm(t_as - t_lm)
    end
    @test ΔTS < 1e-2

    # compare Heisenberg energies
    H = square_lattice_heisenberg(; unitcell)
    E_leftmoves = costfun(psi, env_leftmoves, H)
    E_allsides = costfun(psi, env_allsides, H)
    @test E_leftmoves ≈ E_allsides rtol=1e-4
end
