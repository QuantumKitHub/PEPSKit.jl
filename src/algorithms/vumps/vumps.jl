@kwdef mutable struct VUMPS{F}
    ifupdown::Bool = true
    tol::Float64 = Defaults.contr_tol
    maxiter::Int = Defaults.contr_maxiter
    miniter::Int = Defaults.contr_miniter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity
end

VUMPSRuntime(ipeps::InfinitePEPS, χ::Int) = VUMPSRuntime(ipeps, ℂ^χ)
function VUMPSRuntime(ipeps::InfinitePEPS, χ::VectorSpace)
    A = initial_A(ipeps, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    
    _, FL = leftenv(AL, adjoint.(AL), ipeps)
    _, FR = rightenv(AR, adjoint.(AR), ipeps)
    C = LRtoC(L, R)
    return VUMPSRuntime(AL, AR, C, FL, FR)
end
@non_differentiable VUMPSRuntime(ipeps::InfinitePEPS, χ::VectorSpace)

function _down_ipeps(ipeps::InfinitePEPS)
    Ni, Nj = size(ipeps)
    ipepsd = Zygote.Buffer(ipeps.A)
    for j in 1:Nj, i in 1:Ni
        ir = Ni + 1 - i
        Ad = permute(ipeps[ir,j]', ((5,), (3,2,1,4)))
        ipepsd[i, j] = _fit_spaces(Ad, ipeps[ir,j])
    end
    
    return InfinitePEPS(copy(ipepsd))
end

@non_differentiable VUMPSRuntime(ipeps::InfinitePEPS, χ::VectorSpace, alg::VUMPS)
function VUMPSRuntime(ipeps::InfinitePEPS, χ::VectorSpace, alg::VUMPS)
    Ni, Nj = size(ipeps)

    rtup = VUMPSRuntime(ipeps, χ)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown
        ipepsd = _down_ipeps(ipeps)
        rtdown = VUMPSRuntime(ipepsd, χ)
        alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"

        return rtup, rtdown
    else
        return rtup
    end
end

function vumps_itr(rt::VUMPSRuntime, ipeps::InfinitePEPS, alg::VUMPS)
    t = Zygote.@ignore time()
    for i in 1:alg.maxiter
        rt, err = vumps_step(rt, ipeps, alg)
        alg.verbosity >= 3 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol && i >= alg.miniter
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end
    return rt
end

function leading_boundary(rt::VUMPSRuntime, ipeps::InfinitePEPS, alg::VUMPS)
    return vumps_itr(rt, ipeps, alg)
end

function VUMPSEnv(rt::VUMPSRuntime)
    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FL, FR)
end

function leading_boundary(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, ipeps::InfinitePEPS, alg::VUMPS)
    rtup, rtdown = rt
    
    rtup = vumps_itr(rtup, ipeps, alg)

    ipepsd = _down_ipeps(ipeps)
    rtdown = vumps_itr(rtdown, ipepsd, alg)

    return rtup, rtdown
end

function VUMPSEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}, ipeps::InfinitePEPS)
    rtup, rtdown = rt

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    # to do fix the follow index
    _, FLo =  leftenv(ALu, adjoint.(ALd), ipeps, FLu; ifobs = true)
    _, FRo = rightenv(ARu, adjoint.(ARd), ipeps, FRu; ifobs = true)

    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

function vumps_step(rt::VUMPSRuntime, ipeps::InfinitePEPS, alg::VUMPS)
    verbosity = alg.verbosity
    @unpack AL, C, AR, FL, FR = rt
    AC = Zygote.@ignore ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, FR, ipeps; verbosity)
    _,  Cp =  Cenv( C, FL, FR; verbosity)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL =  leftenv(AL, adjoint.(ALp), ipeps, FL; verbosity)
    _, FR = rightenv(AR, adjoint.(ARp), ipeps, FR; verbosity)
    _, ACp = ACenv(ACp, FL, FR, ipeps; verbosity)
    _,  Cp =  Cenv( Cp, FL, FR; verbosity)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")

    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end