@kwdef mutable struct VUMPS{F} <: Algorithm
    ifupdown::Bool = true
    tol::Float64 = Defaults.contr_tol
    maxiter::Int = Defaults.contr_maxiter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity
end

"""
    VUMPSEnv{T<:Number, S<:IndexSpace,
             OT<:AbstractTensorMap{S, 2, 2},
             ET<:AbstractTensorMap{S, 2, 1},
             CT<:AbstractTensorMap{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for calculate observables.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `AC`: The mixed canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `Lu`: The left upper environment tensor.
- `Ru`: The right upper environment tensor.
- `Lo`: The left mixed environment tensor.
- `Ro`: The right mixed environment tensor.
"""
struct VUMPSEnv{T<:Number, S<:IndexSpace,
                ET<:AbstractTensorMap{S, 3, 1}}
    ACu::Matrix{ET}
    ARu::Matrix{ET}
    ACd::Matrix{ET}
    ARd::Matrix{ET}
    FLu::Matrix{ET}
    FRu::Matrix{ET}
    FLo::Matrix{ET}
    FRo::Matrix{ET}
    function VUMPSEnv(ACu::Matrix{ET},
                      ARu::Matrix{ET},
                      ACd::Matrix{ET},
                      ARd::Matrix{ET},
                      FLu::Matrix{ET},
                      FRu::Matrix{ET},
                      FLo::Matrix{ET},
                      FRo::Matrix{ET}) where {ET}
        T = eltype(ACu[1])
        S = spacetype(ACu[1])
        new{T, S, ET}(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
    end
end

"""
    VUMPSRuntime{T<:Number, S<:IndexSpace,
                 OT<:AbstractTensorMap{S, 2, 2},
                 ET<:AbstractTensorMap{S, 2, 1},
                 CT<:AbstractTensorMap{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for runtime calculations.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `O`: The center transfer matrix PEPO tensor.
- `AL`: The left canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `C`: The canonical environment tensor.
- `L`: The left environment tensor.
- `R`: The right environment tensor.
"""
struct VUMPSRuntime{T<:Number, S<:IndexSpace,
                    ET<:AbstractTensorMap{S, 3, 1},
                    CT<:AbstractTensorMap{S, 1, 1}}
    AL::Matrix{ET}
    AR::Matrix{ET}
    C::Matrix{CT}
    FL::Matrix{ET}
    FR::Matrix{ET}
    function VUMPSRuntime(AL::Matrix{ET},
                          AR::Matrix{ET},
                          C::Matrix{CT},
                          FL::Matrix{ET},
                          FR::Matrix{ET}) where {ET, CT}
        T = eltype(AL[1])
        S = spacetype(AL[1])
        new{T, S, ET, CT}(AL, AR, C, FL, FR)
    end
end

@non_differentiable VUMPSRuntime(O::InfiniteTransferPEPS, χ::VectorSpace)
function VUMPSRuntime(O::InfiniteTransferPEPS, χ::VectorSpace)
    A = initial_A(O, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    
    _, FL = leftenv(AL, adjoint.(AL), O)
    _, FR = rightenv(AR, adjoint.(AR), O)
    C = LRtoC(L, R)
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

function down_itp(O)
    Ni, Nj = size(O)
    ipepsd = Zygote.Buffer(O.top)
    for j in 1:Nj, i in 1:Ni
        ir = Ni + 1 - i 
        Ad = permute(O.top[ir,j]', ((5,), (3,2,1,4)))
        ipepsd[i, j] = _fit_spaces(Ad, O.top[ir,j])
    end
    
    return InfiniteTransferPEPS(copy(ipepsd))
end

@non_differentiable VUMPSRuntime(O::InfiniteTransferPEPS, χ::VectorSpace, alg::VUMPS)
function VUMPSRuntime(O::InfiniteTransferPEPS, χ::VectorSpace, alg::VUMPS)
    Ni, Nj = size(O)

    rtup = VUMPSRuntime(O, χ)
    alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) up(↑) environment"

    if alg.ifupdown
        Od = down_itp(O)
        rtdown = VUMPSRuntime(Od, χ)
        alg.verbosity >= 2 && Zygote.@ignore @info "VUMPS init: cell=($(Ni)×$(Nj)) χ = $(χ) down(↓) environment"

        return rtup, rtdown
    else
        return rtup
    end
end

function vumps_itr(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    t = Zygote.@ignore time()
    for i in 1:alg.maxiter
        rt, err = vumps_step(O, rt, alg)
        alg.verbosity >= 3 && Zygote.@ignore @info @sprintf("VUMPS@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        if err < alg.tol
            alg.verbosity >= 2 && Zygote.@ignore @info @sprintf("VUMPS conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && Zygote.@ignore @warn @sprintf("VUMPS cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
        end
    end
    return rt
end

function vumps(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    return vumps_itr(O, rt, alg)
end

function leading_boundary(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    rt = vumps(O, rt, alg)

    @unpack AL, AR, C, FL, FR = rt
    AC = ALCtoAC(AL, C)
    return VUMPSEnv(AC, AR, AC, AR, FL, FR, FL, FR)
end

function vumps(O::InfiniteTransferPEPS, rt::Tuple, alg::VUMPS)
    rtup, rtdown = rt

    rtup = vumps_itr(O, rtup, alg)

    Od = down_itp(O)
    rtdown = vumps_itr(Od, rtdown, alg)

    return rtup, rtdown
end

function leading_boundary(O::InfiniteTransferPEPS, rt::Tuple, alg::VUMPS)
    rtup, rtdown = vumps(O, rt, alg)

    ALu, ARu, Cu, FLu, FRu = rtup.AL, rtup.AR, rtup.C, rtup.FL, rtup.FR
    ACu = ALCtoAC(ALu, Cu)

    ALd, ARd, Cd = rtdown.AL, rtdown.AR, rtdown.C
    ACd = ALCtoAC(ALd, Cd)

    # to do fix the follow index
    _, FLo =  leftenv(ALu, adjoint.(ALd), O, FLu; ifobs = true)
    _, FRo = rightenv(ARu, adjoint.(ARd), O, FRu; ifobs = true)

    return VUMPSEnv(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
end

function vumps_step(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    verbosity = alg.verbosity
    @unpack AL, C, AR, FL, FR = rt
    AC = Zygote.@ignore ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, FR, O; verbosity)
    _,  Cp =  Cenv( C, FL, FR; verbosity)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL =  leftenv(AL, adjoint.(ALp), O, FL; verbosity)
    _, FR = rightenv(AR, adjoint.(ARp), O, FR; verbosity)
    _, ACp = ACenv(ACp, FL, FR, O; verbosity)
    _,  Cp =  Cenv( Cp, FL, FR; verbosity)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")

    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end