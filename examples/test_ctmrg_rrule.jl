using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit, Zygote
using PEPSKit, KrylovKit, VectorInterface

# Square lattice Heisenberg Hamiltonian
# Sublattice-rotate to get (1, 1, 1) → (-1, 1, -1), transformed to GS with single-site unit cell
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    Sx, Sy, Sz, _ = spinmatrices(1//2)
    Vphys = ℂ^2
    σx = TensorMap(Sx, Vphys, Vphys)
    σy = TensorMap(Sy, Vphys, Vphys)
    σz = TensorMap(Sz, Vphys, Vphys)

    @tensor H[-1 -3; -2 -4] :=
        Jx * σx[-1, -2] * σx[-3, -4] +
        Jy * σy[-1, -2] * σy[-3, -4] +
        Jz * σz[-1, -2] * σz[-3, -4]

    return NLocalOperator{NearestNeighbor}(H)
end

# Parameters
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)

# Cost function for 'non-builtin' optimization approach

# grad_mode = KrylovKit.GMRES(; tol=ctmalg.tol)
grad_mode = GeomSum()
# grad_mode = ManualIter()
function cfun(x)
    (ψ, envs) = x

    function fun(peps)
        envs′ = leading_boundary(peps, ctmalg, envs; grad_mode) # dummy grad_mode kwarg to be used in pullback
        return PEPSKit.costfun(ψ, envs′, H)
    end

    E, g = withgradient(fun, ψ)
    ∂E∂A = g[1]
    if !(∂E∂A isa InfinitePEPS)
        ∂E∂A = InfinitePEPS(∂E∂A.A)
    end
    @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A
end

# Ground state search
ψ₀ = InfinitePEPS(2, χbond)
env₀ = leading_boundary(ψ₀, ctmalg, CTMRGEnv(ψ₀; Venv=ℂ^χenv))

x, f, normgrad = optimize(
    cfun,
    (ψ₀, env₀),
    LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2); # ConjugateGradient(; gradtol=1e-6, verbosity=4, maxiter=100),
    inner=PEPSKit.my_inner,
    retract=PEPSKit.my_retract,
    (scale!)=VectorInterface.scale!,
    (add!)=VectorInterface.add!,
)

@show f
