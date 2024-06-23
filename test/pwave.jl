using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

"""
    square_lattice_pwave(; t=1, μ=2, Δ=1)

    Square lattice p-wave superconductor model.
"""
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

# Initialize parameters
H = square_lattice_pwave()
χbond = 2
χenv = 24
ctm_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=400, fixedspace=true, verbosity=1
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=10),
    reuse_env=true,
    verbosity=2,
)

# initialize states
Random.seed!(91283219347)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Vspace = Vect[FermionParity](0 => χbond ÷ 2, 1 => χbond ÷ 2)
Envspace = Vect[FermionParity](0 => χenv ÷ 2, 1 => χenv ÷ 2)
psi_init = InfinitePEPS(Pspace, Vspace, Vspace)
env_init = leading_boundary(CTMRGEnv(psi_init; Venv=Envspace), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

@test result.E ≈ -2.60053 atol = 1e-2 #comparison with Gaussian PEPS minimum at D=2 on 1000x1000 square lattice with aPBC
