using Test
using Random
using TensorKit
using KrylovKit
using LineSearches, Manifolds, Manopt
using PEPSKit

##
using KrylovKit
using TensorKit
using PEPSKit

Dbond = 2
χenv = 16
svd_alg = SVDAdjoint(; fwd_alg=IterSVD(; alg=GKL(; tol=1e-10)))
ctmrg_alg = SimultaneousCTMRG(; svd_alg)
ψ = InfinitePEPS(2, Dbond)
env = leading_boundary(CTMRGEnv(ψ, ℂ^χenv), ψ, ctmrg_alg);

##
# initialize parameters
Dbond = 2
χenv = 16
# ctm_alg = SequentialCTMRG()
# opt_alg = PEPSOptimize(; boundary_alg=ctm_alg, tol=1e-3, gradient_alg=LinSolver(; solver=GMRES(; tol=1e-4), iterscheme=:diffgauge))
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(; boundary_alg=ctm_alg, tol=1e-3)
# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
E_ref = -0.6602310934799577

# initialize states
Random.seed!(123)
H = heisenberg_XYZ(InfiniteSquare())
psi_init = InfinitePEPS(2, Dbond)
env_init, = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

# optimize energy and compute correlation lengths
peps, env, E, = fixedpoint(psi_init, H, opt_alg, env_init);
ξ_h, ξ_v, = correlation_length(peps, env)

@test E ≈ E_ref atol = 1e-2
@test all(@. ξ_h > 0 && ξ_v > 0)

##
g = 3.1
e = -1.6417 * 2
mˣ = 0.91

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    tol=1e-3,
    # stepsize=WolfePowellBinaryLinesearch(),
    # stepsize=Manopt.LineSearchesStepsize(LineSearches.HagerZhang(; alphamax=1.0)),
    # stepsize=ConstantLength(1.0),
    # stepsize=WolfePowellLinesearch(),
    # stepsize=WolfePowellLinesearch(; sufficient_decrease=0.1, sufficient_curvature=0.9),
    # direction_update=InverseBroyden(1.0),
    # cautious_update=true,
)

# initialize states
H = transverse_field_ising(InfiniteSquare(); g)
H_polar = transverse_field_ising(InfiniteSquare(); g=4.5)
Random.seed!(91283219347)
psi_init = InfinitePEPS(2, χbond)
env_init, = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

# find fixedpoint
peps, env, E, = fixedpoint(psi_init, H, opt_alg, env_init);
# peps_polar, env_polar, E_polar, = fixedpoint(psi_init, H_polar, opt_alg, env_init);

##
χbond = 2
χenv = 12
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    tol=1e-3,
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
    symmetrization=RotateReflect(),
)

# initialize states
Random.seed!(91283219347)
H = j1_j2(InfiniteSquare(); J2=0.25)
psi_init = product_peps(2, χbond; noise_amp=1e-1)
psi_init = symmetrize!(psi_init, RotateReflect())
env_init, = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

##
unitcell = (2, 2)
H = pwave_superconductor(InfiniteSquare(unitcell...))
Dbond = 2
χenv = 16
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    maxiter=10,
    gradient_alg=LinSolver(; iterscheme=:diffgauge),
    # stepsize=WolfePowellLinesearch(),
    stepsize=ConstantLength(1.5),
    memory_size=4,
)

# initialize states
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Vspace = Vect[FermionParity](0 => Dbond ÷ 2, 1 => Dbond ÷ 2)
Envspace = Vect[FermionParity](0 => χenv ÷ 2, 1 => χenv ÷ 2)
Random.seed!(91283219347)
psi_init = InfinitePEPS(Pspace, Vspace, Vspace; unitcell)
env_init, = leading_boundary(CTMRGEnv(psi_init, Envspace), psi_init, ctm_alg);

# find fixedpoint
peps, env, E, = fixedpoint(psi_init, H, opt_alg, env_init);
@show E / 4

# comparison with Gaussian PEPS minimum at D=2 on 1000x1000 square lattice with aPBC
@test E / prod(size(psi_init)) ≈ -2.6241 atol = 5e-2
