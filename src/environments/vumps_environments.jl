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
    Lu::Matrix{ET}
    Ru::Matrix{ET}
    Lo::Matrix{ET}
    Ro::Matrix{ET}
    function VUMPSEnv(AC::Matrix{ET},
                      AR::Matrix{ET},
                      Lu::Matrix{ET},
                      Ru::Matrix{ET},
                      Lo::Matrix{ET},
                      Ro::Matrix{ET}) where {ET}
        T = eltype(AC[1])
        S = spacetype(AC[1])
        new{T, S, ET}(AC, AR, Lu, Ru, Lo, Ro)
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
                    OT<:InfiniteTransferPEPS,
                    ET<:AbstractTensorMap{S, 3, 1},
                    CT<:AbstractTensorMap{S, 1, 1}}
    O::OT
    AL::Matrix{ET}
    AR::Matrix{ET}
    C::Matrix{CT}
    L::Matrix{ET}
    R::Matrix{ET}
    function VUMPSRuntime(O::OT,
                          AL::Matrix{ET},
                          AR::Matrix{ET},
                          C::Matrix{CT},
                          L::Matrix{ET},
                          R::Matrix{ET}) where {OT, ET, CT}
        T = eltype(AL[1])
        S = spacetype(AL[1])
        new{T, S, OT, ET, CT}(O, AL, AR, C, L, R)
    end
end

"""

````

    l ←------- r
        / \
       /   \
      t     d
````
Initalize a boundary MPS for the transfer operator `O` by specifying an array of virtual
spaces consistent with the unit cell.
"""
function initial_A(O::InfiniteTransferPEPS, χ::VectorSpace)
    T = eltype(O)
    Ni, Nj = size(O)
    A = Matrix{TensorMap}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        D = space(O.top[i, j], 4)
        A[i, j] = TensorMap(rand, T, χ * D * D', χ)
    end
    return A
end

function initial_C(A::Matrix{<:AbstractTensorMap})
    T = eltype(A[1])
    Ni, Nj = size(A)
    C = Matrix{TensorMap}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        χ = space(A[i, j], 1) 
        C[i, j] = isomorphism(Matrix{T}, χ, χ) # only for CPU
    end
    return C
end

function VUMPSRuntime(ψ₀, χ::Int)
    
end