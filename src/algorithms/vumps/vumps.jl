@kwdef mutable struct VUMPS{F} <: Algorithm
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
    AC::Matrix{ET}
    AR::Matrix{ET}
    FLu::Matrix{ET}
    FRu::Matrix{ET}
    FLo::Matrix{ET}
    FRo::Matrix{ET}
    function VUMPSEnv(AC::Matrix{ET},
                      AR::Matrix{ET},
                      FLu::Matrix{ET},
                      FRu::Matrix{ET},
                      FLo::Matrix{ET},
                      FRo::Matrix{ET}) where {ET}
        T = eltype(AC[1])
        S = spacetype(AC[1])
        new{T, S, ET}(AC, AR, FLu, FRu, FLo, FRo)
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

function VUMPSRuntime(O::InfiniteTransferPEPS, χ::VectorSpace, alg::VUMPS)
    A = initial_A(O, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    
    _, FL = leftenv(AL, adjoint.(AL), O)
    _, FR = rightenv(AR, adjoint.(AR), O)
    C = LRtoC(L, R)
    Ni, Nj = size(O)
    alg.verbosity >= 2 && println("===== vumps random initial $(Ni)×$(Nj) vumps χ = $(χ) environment  =====")
    return VUMPSRuntime(AL, AR, C, FL, FR)
end

function leading_boundary(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    for i in 1:alg.maxiter
        rt, err = vumps_itr(O, rt, alg)
        alg.verbosity >= 3 && println(@sprintf("vumps@step: i = %4d\terr = %.3e\t", i, err))
        if err < alg.tol
            alg.verbosity >= 2 && println(@sprintf("===== vumps@step: i = %4d\terr = %.3e\t coveraged =====", i, err))
            break
        end
        if i == alg.maxiter
            alg.verbosity >= 2 && println(@sprintf("===== vumps@step: i = %4d\terr = %.3e\t not coveraged =====", i, err))
        end
    end
    return rt
end

function vumps_itr(O::InfiniteTransferPEPS, rt::VUMPSRuntime, alg::VUMPS)
    @unpack AL, C, AR, FL, FR = rt
    AC = Zygote.@ignore ALCtoAC(AL,C)
    _, ACp = ACenv(AC, FL, FR, O)
    _,  Cp =  Cenv( C, FL, FR)
    ALp, ARp, _, _ = ACCtoALAR(ACp, Cp)
    _, FL =  leftenv(AL, adjoint.(ALp), O, FL)
    _, FR = rightenv(AR, adjoint.(ARp), O, FR)
    _, ACp = ACenv(ACp, FL, FR, O)
    _,  Cp =  Cenv( Cp, FL, FR)
    ALp, ARp, errL, errR = ACCtoALAR(ACp, Cp)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")

    return VUMPSRuntime(ALp, ARp, Cp, FL, FR), err
end