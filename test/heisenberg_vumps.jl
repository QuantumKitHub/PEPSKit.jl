using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using Printf

@testset "heisenberg_XYZ 1x1 unitcell C4 symmetry" begin
    Random.seed!(100)
    # initialize parameters
    χbond = 2
    χenv = 10

    # initialize states
    H = heisenberg_XYZ(InfiniteSquare())
    psi_init = InfinitePEPS(2, χbond; unitcell=(1, 1))

    # find fixedpoint one-site ctmrg
    boundary_alg = VUMPS(
        ifupdown=false,
        tol=1e-10,
        miniter=3,
        maxiter=10,
        verbosity=1
    )
    opt_alg = PEPSOptimize(;
        boundary_alg,
        optimizer=LBFGS(20; maxiter=100, gradtol=1e-6, verbosity=2),
        gradient_alg=nothing,
        reuse_env=true
    )

    t = time() 
    function finalize!(x, f, g, numiter)
        print(@sprintf("%.3f sec", time()-t))
        return x, f, g
    end

    env_init = VUMPSRuntime(psi_init, χenv, boundary_alg)
    result = fixedpoint(psi_init, H, opt_alg, env_init; 
                        symmetrization=RotateReflect(),
                        finalize!
                        );
    @test result.E ≈ -0.66023 atol = 1e-4
end

@testset "heisenberg_XYZ 1x1 unitcell without C4 symmetry" begin
    Random.seed!(100)
    # initialize parameters
    χbond = 2
    χenv = 10

    # initialize states
    H = heisenberg_XYZ(InfiniteSquare())
    psi_init = InfinitePEPS(2, χbond; unitcell=(1, 1))

    # find fixedpoint one-site ctmrg
    boundary_alg = VUMPS(
        ifupdown=true,
        tol=1e-10,
        miniter=1,
        maxiter=10,
        verbosity=1
    )
    opt_alg = PEPSOptimize(;
        boundary_alg,
        optimizer=LBFGS(20; maxiter=100, gradtol=1e-6, verbosity=2),
        gradient_alg=nothing,
        reuse_env=true
    )

    t = time() 
    function finalize!(x, f, g, numiter)
        print(@sprintf("%.3f sec", time()-t))
        return x, f, g
    end

    env_init = VUMPSRuntime(psi_init, χenv, boundary_alg)
    result = fixedpoint(psi_init, H, opt_alg, env_init; 
                        finalize!
                        );
    @test result.E ≈ -0.66251 atol = 1e-4
end

@testset "heisenberg_XYZ 2x2 unitcell without C4 symmetry" begin
    Random.seed!(42)
    # initialize parameters
    χbond = 2
    χenv = 16

    # initialize states
    H = heisenberg_XYZ(InfiniteSquare())
    psi_init = InfinitePEPS(2, χbond; unitcell=(2, 2))

    # find fixedpoint one-site ctmrg
    boundary_alg = VUMPS(
        ifupdown=true,
        tol=1e-10,
        miniter=1,
        maxiter=10,
        verbosity=2
    )
    opt_alg = PEPSOptimize(;
        boundary_alg,
        optimizer=LBFGS(20; maxiter=100, gradtol=1e-6, verbosity=2),
        gradient_alg=nothing,
        reuse_env=true
    )

    t = time() 
    function finalize!(x, f, g, numiter)
        print(@sprintf("%.3f sec", time()-t))
        return x, f, g
    end

    env_init = VUMPSRuntime(psi_init, χenv, boundary_alg)
    result = fixedpoint(psi_init, H, opt_alg, env_init; 
                        finalize!
    );
    @show result.E
    # @test result.E ≈ -0.66251 atol = 1e-4
end
