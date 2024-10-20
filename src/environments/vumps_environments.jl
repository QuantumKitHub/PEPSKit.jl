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
                OT<:AbstractTensorMap{S, 2, 2},
                ET<:AbstractTensorMap{S, 2, 1},
                CT<:AbstractTensorMap{S, 1, 1}}
    AC::Matrix{ET}
    AR::Matrix{ET}
    Lu::Matrix{ET}
    Ru::Matrix{ET}
    Lo::Matrix{ET}
    Ro::Matrix{ET}
    function VUMPSEnv(AC::Matrix{ET},
                      AR::Matrix{ET}
                      Lu::Matrix{ET},
                      Ru::Matrix{ET},
                      Lo::Matrix{ET},
                      Ro::Matrix{ET}) where {OT, ET, CT}
        T = eltype(O[1])
        new{T, S, OT, ET, CT}(AC, AR, Lu, Ru, Lo, Ro)
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
                    OT<:AbstractTensorMap{S, 2, 2},
                    ET<:AbstractTensorMap{S, 2, 1},
                    CT<:AbstractTensorMap{S, 1, 1}}
    O::Matrix{OT}
    AL::Matrix{ET}
    AR::Matrix{ET}
    C::Matrix{CT}
    L::Matrix{ET}
    R::Matrix{ET}
    function VUMPSRuntime(O::Matrix{OT},
                          AL::Matrix{ET},
                          AR::Matrix{ET},
                          C::Matrix{CT},
                          L::Matrix{ET},
                          R::Matrix{ET}) where {OT, ET, CT}
        T = eltype(O[1])
        new{T, S, OT, ET, CT}(O, AL, AR, C, L, R)
    end
end

function VUMPSRuntime(ψ₀, χ::Int)
    
end