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

# KrylovKit patch
TensorKit.inner(x::Matrix{TensorMap}, y::Matrix{TensorMap}) = sum(map(TensorKit.inner, x, y))
TensorKit.add!!(x::Matrix{<:AbstractTensorMap}, y::Matrix{<:AbstractTensorMap}, a::Number, b::Number) = map((x, y) -> TensorKit.add!!(x, y, a, b), x, y)
TensorKit.scale!!(x::Matrix{<:AbstractTensorMap}, a::Number) = map(x -> TensorKit.scale!!(x, a), x)

"""
    λs[1], Fs[1] = selectpos(λs, Fs)

Select the max positive one of λs and corresponding Fs.
"""
function selectpos(λs, Fs, N)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        # @show "selectpos: λs are degeneracy"
        N = min(N, length(λs))
        p = argmax(real(λs[1:N]))  
        # @show λs p abs.(λs)
        return λs[1:N][p], Fs[1:N][p]
    else
        return λs[1], Fs[1]
    end
end

"""
    L = getL!(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap}; verbosity = Defaults.verbosity, kwargs...)

````
     ┌─ Aᵢⱼ ─ Aᵢⱼ₊₁─     ┌─      L ─
     ρᵢⱼ │      │     =  ρᵢⱼ  =  │
     └─ Aᵢⱼ─  Aᵢⱼ₊₁─     └─      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.

"""
function getL!(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap}; verbosity = Defaults.verbosity, kwargs...)
    Ni, Nj = size(A)
    λs, ρs, info = eigsolve(ρ -> ρmap(ρ, A), L, 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
    verbosity >= 1 && info.converged == 0 && @warn "getL not converged"
    _, ρs1 = selectpos(λs, ρs, Nj)
    @inbounds for j = 1:Nj, i = 1:Ni
        ρ = ρs1[i,j] + ρs1[i,j]'
        ρ /= tr(ρ)
        _, S, Vt = tsvd!(ρ)
        # Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
        Lo = sqrt(S) * Vt
        _, R = leftorth!(Lo)
        L[i,j] = R
    end
    return L
end

"""
    AL, Le, λ = getAL(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap})

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ * AL * Le = L * A``
"""
function getAL(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap})
    Ni, Nj = size(A)
    AL = similar(A)
    Le = similar(L)
    λ = zeros(Ni, Nj)
    @inbounds for j = 1:Nj, i = 1:Ni
        Q, R = leftorth!(transpose(L[i,j]*transpose(A[i,j], ((1,),(4,3,2))), ((1,4,3),(2,))))
        AL[i,j] = Q
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le::Matrix{<:AbstractTensorMap}, A::Matrix{<:AbstractTensorMap}, AL::Matrix{<:AbstractTensorMap}; verbosity = Defaults.verbosity, kwargs...)
    Ni, Nj = size(A)
    L = similar(Le)
    @inbounds for j = 1:Nj, i = 1:Ni
        λs, Ls, info = eigsolve(L -> (@tensor Ln[-1; -2] := L[4; 1] * A[i,j][1 2 3; -2] * conj(AL[i,j][4 2 3; -1])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "getLsped not converged"
        _, Ls1 = selectpos(λs, Ls, Nj)
        _, R = leftorth!(Ls1)
        L[i,j] = R
    end
    return L
end

"""
    AL, L, λ = left_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ*AL*L = L*A``, where an initial guess for `L` can be
provided.
"""
function left_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)
    L = getL!(A, L; kwargs...)
    AL, Le, λ = getAL(A, L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end

"""
    R, AR, λ = right_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ * R * AR = A * R``, where an initial guess for `R` can be
provided.
"""
function right_canonical(A::Matrix{<:AbstractTensorMap}, L::Matrix{<:AbstractTensorMap} = initial_C(A); tol = 1e-12, maxiter = 100, kwargs...)
    Ar = [permute(A, ((4,2,3), (1,))) for A in A]
    Lr = [permute(L, ((2,   ), (1,))) for L in L]
    
    AL, L, λ = left_canonical(Ar, Lr; tol, maxiter, kwargs...)

     R = [permute( L, ((2,), (1,   ))) for  L in  L]
    AR = [permute(AL, ((4,), (2,3,1))) for AL in AL]
    return R, AR, λ
end

function initial_FL(AL::Matrix{<:AbstractTensorMap}, O::InfiniteTransferPEPS)
    T = eltype(O)
    Ni, Nj = size(O)
    FL = Matrix{TensorMap}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        D = space(O.top[i, j], 5)
        χ = space(AL[i, j], 1)
        FL[i, j] = TensorMap(rand, T, χ' * D * D', χ)
    end
    
    return FL
end

function initial_FR(AR::Matrix{<:AbstractTensorMap}, O::InfiniteTransferPEPS)
    T = eltype(O)
    Ni, Nj = size(O)
    FR = Matrix{TensorMap}(undef, Ni, Nj)
    for j in 1:Nj, i in 1:Ni
        D = space(O.top[i, j], 3)
        χ = space(AR[i, j], 1)
        FR[i, j] = TensorMap(rand, T, χ * D * D', χ')
    end
    
    return FR
end

function VUMPSRuntime(ψ₀, χ::Int)
    
end