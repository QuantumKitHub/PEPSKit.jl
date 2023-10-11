# this should run once the compatocalypse has been resolved

using Revise
using TensorOperations
using TensorKit
using MPSKit
using PEPSKit
using OptimKit


## Set some options

optim_method = ConjugateGradient
optim_tol = 1e-5
optim_maxiter = 100
verbosity = 2 # >= 5 for debugging
boundary_maxiter = 50
tol_min = 1e-12
tol_max = 1e-4
tol_factor = 1e-3
symm = Full()
hermitian = true
boundary_method = VUMPS(; tol_galerkin=tol_max, maxiter=boundary_maxiter, dynamical_tols=true, eigs_tolfactor=1e-3, envs_tolfactor=1e-3, gauge_tolfactor=1e-6, tol_max=1e-4, verbose=verbosity >= 5)
# boundary_method = GradientGrassmann


## Set model parameters

# 3D Ising temperature
beta = 1/3 # should give f = -1.0219139... @ D = 3

# bond dimensions
D = 3
χ = 30

# pick size of unit cell
depth = 1
width = 1
height = 1


## PEPO defintion: 3D classical Ising model

t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
q = sqrt(t)

O = zeros(2, 2, 2, 2, 2, 2)
O[1, 1, 1, 1, 1, 1] = 1
O[2, 2, 2, 2, 2, 2] = 1
@tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

M = zeros(2, 2, 2, 2, 2, 2)
M[1, 1, 1, 1, 1, 1] = 1
M[2, 2, 2, 2, 2, 2] = -1
@tensor m[-1 -2; -3 -4 -5 -6] := M[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

o = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')
m = TensorMap(m, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

O = InfinitePEPO(repeat([o], depth, width, height))


## Optimization algorithm

pepo_alg = PEPOOptimize(;
    optim_method,
    optim_tol,
    optim_maxiter,
    verbosity,
    boundary_method,
    boundary_maxiter,
    tol_min,
    tol_max,
    tol_factor,
    symm,
    hermitian,
)


## Initialize PEPS and MPS fixed points

peps = symmetrize(initializePEPS(O, ℂ^D), symm)
envs = pepo_opt_environments(peps, O, pepo_alg.boundary_method; vspaces=[ℂ^χ], hermitian=pepo_alg.hermitian)
normalize!(peps, envs.peps_boundary)


## Perform optimization

x, f, normgrad = leading_boundary(peps, O, pepo_alg, envs)


## Unpack result and check magnetization

peps = x.state
above = x.envs.pepo_boundary.boundaries[1]
below = x.envs.pepo_boundary.boundaries[2]
lw, rw = x.envs.pepo_boundary.envs[3]

pos = (1, 1) # pick a site

if height == 1 # no crazy contractions for now...
    row, col = pos
    fliprow = size(peps, 1) - row + 1 # below starts counting from below
    m = @tensor lw[row, col][13 8 10 18; 1] *
        above.AC[row, col][1 9 11 15; 12] *
        peps[row, col][5; 9 3 4 8] *
        m[14 5; 11 6 7 10] *
        rw[row, col][12 3 6 16; 2] *
        conj(below.AC[fliprow, col][13 4 7 17; 2]) *
        conj(peps[row, col][14; 15 16 17 18])
    
    λ = @tensor lw[row, col][13 8 10 18; 1] *
        above.AC[row, col][1 9 11 15; 12] *
        peps[row, col][5; 9 3 4 8] *
        t[14 5; 11 6 7 10] *
        rw[row, col][12 3 6 16; 2] *
        conj(below.AC[fliprow, col][13 4 7 17; 2]) *
        conj(peps[row, col][14; 15 16 17 18])
    
    println("m = $(m / λ)") # should be ~0.945 @ beta = 1/3 and D = 3
end

nothing
