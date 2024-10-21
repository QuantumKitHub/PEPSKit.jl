@kwdef mutable struct VUMPS{F} <: Algorithm
    tol::Float64 = Defaults.contr_tol
    maxiter::Int = Defaults.contr_maxiter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity
end

function leading_boundary(O::Matrix, alg::VUMPS)
    # initialize
    rt = VUMPSRuntime(O, nothing, nothing, nothing, nothing, nothing)
    return rt
end

function vumps(rt::VUMPSRuntime; tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose=false, show_every = Inf)
    # initialize
    olderror = Inf
    vumps_counting = show_every_count(show_every)

    stopfun = StopFunction(olderror, -1, tol, maxiter, miniter)
    rt, err = fixedpoint(res -> vumpstep(res...;show_counting=vumps_counting), (rt, olderror), stopfun)
    verbose && println("vumps done@step: $(stopfun.counter), error=$(err)")
    return rt, err
end