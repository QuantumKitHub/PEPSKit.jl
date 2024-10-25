"""
    InfiniteTransferPEPS{T}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the overlap between 'ket' and 'bra' `InfinitePEPS` states.
"""
struct InfiniteTransferPEPS{TT, BT}
    top::Matrix{TT}
    bot::Matrix{BT}
end

function InfiniteTransferPEPS(ipeps::InfinitePEPS)
    top = ipeps.A
    bot = adjoint.(ipeps.A)
    return InfiniteTransferPEPS(top, bot)
end

function InfiniteTransferPEPS(A::Matrix{<:AbstractTensorMap})
    top = A
    bot = adjoint.(A)
    return InfiniteTransferPEPS(top, bot)
end

function ChainRulesCore.rrule(::Type{InfiniteTransferPEPS}, top::Matrix, bot::Matrix)
    function pullback(Δ)
        return NoTangent(), Δ.top, Δ.bot
    end
    return InfiniteTransferPEPS(top, bot), pullback
end

Base.eltype(transfer::InfiniteTransferPEPS) = eltype(transfer.top[1])
Base.size(transfer::InfiniteTransferPEPS) = size(transfer.top)

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
    A = [(D = space(O.top[i, j], 2)';
         TensorMap(rand, T, χ * D * D', χ)) for i in 1:Ni, j in 1:Nj]
    return A
end

function initial_C(A::Matrix{<:AbstractTensorMap})
    T = eltype(A[1])
    Ni, Nj = size(A)
    C = [(χ = space(A[i, j], 1);
          isomorphism(Matrix{T}, χ, χ)) for i in 1:Ni, j in 1:Nj]
    return C
end

# KrylovKit patch
TensorKit.inner(x::AbstractArray{<:AbstractTensorMap}, y::AbstractArray{<:AbstractTensorMap}) = sum(map(TensorKit.inner, x, y))
TensorKit.add!!(x::AbstractArray{<:AbstractTensorMap}, y::AbstractArray{<:AbstractTensorMap}, a::Number, b::Number) = map((x, y) -> TensorKit.add!!(x, y, a, b), x, y)
TensorKit.scale!!(x::AbstractArray{<:AbstractTensorMap}, a::Number) = map(x -> TensorKit.scale!!(x, a), x)

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

function _to_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1..., reverse(Base.tail(I2))...), (I2[1],)))
end

function _to_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1[1],), (I2..., reverse(Base.tail(I1))...)))
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
        Q, R = leftorth!(_to_front(L[i,j] * _to_tail(A[i,j])))
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
    AR = [permute(AL, ((4,2,3), (1,))) for AL in AL]
    return R, AR, λ
end

function initial_FL(AL::Matrix{<:AbstractTensorMap}, O::InfiniteTransferPEPS)
    T = eltype(O)
    FL = [(D = space(top, 5)';
          χ = space(AL, 4)';
          TensorMap(rand, T, χ * D * D', χ)) for (top, AL) in zip(O.top, AL)]
    return FL
end

function initial_FR(AR::Matrix{<:AbstractTensorMap}, O::InfiniteTransferPEPS)
    T = eltype(O)
    FR = [(D = space(top, 3)';
           χ = space(AR, 4)';
           TensorMap(rand, T, χ * D * D', χ)) for (top, AR) in zip(O.top, AR)]
    return FR
end

"""
    λL, FL = leftenv(ALu, ALd, O, FL = initial_FL(ALu,O); kwargs...)

Compute the left environment tensor for MPS A and MPO O, by finding the left fixed point
of ALu - O - ALd contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ──          ┌── 
 │     │                 │   
FLᵢⱼ ─ Oᵢⱼ   ──   = λLᵢⱼ FLᵢⱼ₊₁   
 │     │                 │   
 └──  ALdᵢᵣⱼ  ─          └── 
```
"""
leftenv(ALu::Matrix{<:AbstractTensorMap}, 
        ALd::Matrix{<:AbstractTensorMap}, 
        O::InfiniteTransferPEPS, 
        FL::Matrix{<:AbstractTensorMap} = initial_FL(ALu,O); 
        kwargs...) = leftenv!(ALu, ALd, O, deepcopy(FL); kwargs...) 

function leftenv!(ALu::Matrix{<:AbstractTensorMap}, 
                  ALd::Matrix{<:AbstractTensorMap}, 
                  O::InfiniteTransferPEPS, 
                  FL::Matrix{<:AbstractTensorMap}; 
                  ifobs=false, verbosity = Defaults.verbosity, kwargs...)

    Ni, Nj = size(O)
    λL = zeros(eltype(O), Ni)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        λLs, FL1s, info = eigsolve(FLi -> FLmap(FLi, ALu[i,:], ALd[ir,:], O.top[i,:], O.bot[i,:]), 
                                   FL[i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
        λL[i], FL[i,:] = selectpos(λLs, FL1s, Nj)
    end
    return λL, FL
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        │       │            │  
    ── Mᵢⱼ   ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ₋₁
        │       │            │  
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
rightenv(ARu::Matrix{<:AbstractTensorMap}, 
         ARd::Matrix{<:AbstractTensorMap}, 
         O::InfiniteTransferPEPS, 
         FR::Matrix{<:AbstractTensorMap} = initial_FR(ARu,O); 
         kwargs...) = rightenv!(ARu, ARd, O, deepcopy(FR); kwargs...) 

function rightenv!(ARu::Matrix{<:AbstractTensorMap},
                   ARd::Matrix{<:AbstractTensorMap}, 
                   O::InfiniteTransferPEPS, 
                   FR::Matrix{<:AbstractTensorMap}; 
                   ifobs=false, ifinline=false,verbosity = Defaults.verbosity, kwargs...) 

    Ni, Nj = size(O)
    λR = zeros(eltype(O), Ni)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        ifinline && (ir = i) 
        λRs, FR1s, info = eigsolve(FR -> FRmap(FR, ARu[i,:], ARd[ir,:], O.top[i,:], O.bot[i,:]), 
                                   FR[i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
        λR[i], FR[i,:] = selectpos(λRs, FR1s, Nj)
    end
    return λR, FR
end

"""
    λR, FR = rightCenv(ARu::Matrix{<:AbstractTensorMap}, 
                       ARd::Matrix{<:AbstractTensorMap}, 
                       R::Matrix{<:AbstractTensorMap} = initial_C(ARu); 
                       kwargs...) 

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        |       Rᵢⱼ  = λRᵢⱼ  Rᵢⱼ₋₁
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
rightCenv(ARu::Matrix{<:AbstractTensorMap}, 
          ARd::Matrix{<:AbstractTensorMap}, 
          R::Matrix{<:AbstractTensorMap} = initial_C(ARu); 
          kwargs...) = rightCenv!(ARu, ARd, deepcopy(R); kwargs...) 

function rightCenv!(ARu::Matrix{<:AbstractTensorMap},
                    ARd::Matrix{<:AbstractTensorMap}, 
                    R::Matrix{<:AbstractTensorMap}; 
                    ifobs=false, verbosity = Defaults.verbosity, kwargs...) 

    Ni, Nj = size(ARu)
    λR = zeros(eltype(ARu[1]), Ni)
    for i in 1:Ni
        ir = ifobs ? mod1(Ni - i + 2, Ni) : i
        λRs, R1s, info = eigsolve(R -> Rmap(R, ARu[i,:], ARd[ir,:]), 
                                   R[i,:], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
        λR[i], R[i,:] = selectpos(λRs, R1s, Nj)
    end
    return λR, R
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐         
│        │         │         =  λACᵢⱼ ┌─── ACᵢ₊₁ⱼ ──┐
FLᵢⱼ ─── Oᵢⱼ ───── FRᵢⱼ               │      │      │   
│        │         │   
```
"""
ACenv(AC::Matrix{<:AbstractTensorMap}, 
      FL::Matrix{<:AbstractTensorMap}, 
      FR::Matrix{<:AbstractTensorMap},
      O::InfiniteTransferPEPS; 
      kwargs...) = ACenv!(deepcopy(AC), FL, FR, O; kwargs...)

function ACenv!(AC::Matrix{<:AbstractTensorMap}, 
                FL::Matrix{<:AbstractTensorMap}, 
                FR::Matrix{<:AbstractTensorMap},
                O::InfiniteTransferPEPS;
                verbosity = Defaults.verbosity, kwargs...)

    Ni, Nj = size(O)
    λAC = zeros(eltype(O),Nj)
    for j in 1:Nj
        λACs, ACs, info = eigsolve(AC -> ACmap(AC, FL[:,j], FR[:,j], O.top[:,j], O.bot[:,j]), 
                                   AC[:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
        λAC[j], AC[:,j] = selectpos(λACs, ACs, Ni)
    end
    return λAC, AC
end

"""
    λC, C = Cenv(C::Matrix{<:AbstractTensorMap}, 
                 FL::Matrix{<:AbstractTensorMap}, 
                 FR::Matrix{<:AbstractTensorMap}; 
                 kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │       =  λCᵢⱼ ┌──Cᵢⱼ ─┐
FLᵢⱼ₊₁ ──── FRᵢⱼ            │       │
│           │   
```
"""
Cenv(C::Matrix{<:AbstractTensorMap}, 
     FL::Matrix{<:AbstractTensorMap}, 
     FR::Matrix{<:AbstractTensorMap}; 
     kwargs...) = Cenv!(copy(C), FL, FR; kwargs...)

function Cenv!(C::Matrix{<:AbstractTensorMap}, 
               FL::Matrix{<:AbstractTensorMap}, 
               FR::Matrix{<:AbstractTensorMap}; 
               verbosity = Defaults.verbosity, kwargs...)

    Ni, Nj = size(C)
    λC = zeros(eltype(C[1]), Nj)
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        λCs, Cs, info = eigsolve(C -> Cmap(C, FL[:,jr], FR[:,j]), 
                                 C[:,j], 1, :LM; maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
        λC[j], C[:,j] = selectpos(λCs, Cs, Ni)
    end
    return λC, C
end

function env_norm(F::Matrix{<:AbstractTensorMap})
    Ni, Nj = size(F)
    buf = Zygote.Buffer(F)
    @inbounds for j in 1:Nj, i in 1:Ni
        buf[i,j] = F[i,j]/norm(F[i,j])
    end
    return copy(buf)
end

"""
    AL, AR = ACCtoALAR(AC, C)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AC::Matrix{<:AbstractTensorMap}, C::Matrix{<:AbstractTensorMap})
    AC = env_norm(AC)
     C = env_norm( C)
    AL, errL = ACCtoAL(AC, C)
    AR, errR = ACCtoAR(AC, C)
    return AL, AR, errL, errR
end

function ACCtoAL(AC::Matrix{<:AbstractTensorMap}, C::Matrix{<:AbstractTensorMap})
    Ni, Nj = size(AC)
    errL = 0.0
    AL = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        QAC, RAC = TensorKit.leftorth(AC[i,j])
         QC, RC  = TensorKit.leftorth( C[i,j])
        errL += norm(RAC - RC)
        AL[i,j] = QAC * QC'
    end
    return copy(AL), errL
end

function ACCtoAR(AC::Matrix{<:AbstractTensorMap}, C::Matrix{<:AbstractTensorMap})
    Ni, Nj = size(AC)
    errR = 0.0
    AR = Zygote.Buffer(AC)
    @inbounds for j in 1:Nj, i in 1:Ni
        jr = j - 1 + (j==1)*Nj
        LAC, QAC = rightorth(_to_tail(AC[i,j]))
         LC, QC  = rightorth(C[i,jr])
        errR += norm(LAC - LC)
        AR[i,j] = _to_front(QC' * QAC)
    end
    return copy(AR), errR
end