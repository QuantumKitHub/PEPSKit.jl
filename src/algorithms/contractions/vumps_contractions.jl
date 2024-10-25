
"""
    ρ = ρmap(ρ::Matrix{<:AbstractTensorMap}, A::Matrix{<:AbstractTensorMap})
````
    ┌─ Aᵢⱼ─    ┌─ 
    ρᵢⱼ │   =  ρⱼ₊₁ 
    └─ Aᵢⱼ─    └─
````
"""
function ρmap(ρ::Matrix{<:AbstractTensorMap}, A::Matrix{<:AbstractTensorMap})
    Ni, Nj = size(ρ)
    ρ = deepcopy(ρ)
    @inbounds for j in 1:Nj, i in 1:Ni
        jr = mod1(j + 1, Nj)
        @tensor ρ[i,jr][-1; -2] = ρ[i,j][4; 1] * A[i,j][1 2 3; -2] * conj(A[i,j][4 2 3; -1]) 
    end
    return ρ
end

"""
    C = LRtoC(L::Matrix{<:AbstractTensorMap}, R::Matrix{<:AbstractTensorMap})

```
    ── Cᵢⱼ ──  =  ── Lᵢⱼ ── Rᵢⱼ₊₁ ──
```
"""
function LRtoC(L::Matrix{<:AbstractTensorMap}, R::Matrix{<:AbstractTensorMap})
    Rijr = circshift(R, (0, -1))
    return L .* Rijr
end

"""
    AC = ALCtoAC(AL::Matrix{<:AbstractTensorMap}, C::Matrix{<:AbstractTensorMap})

```
    ── ACᵢⱼ ──  =  ── ALᵢⱼ ── Cᵢⱼ ──
        |              |
```
"""
function ALCtoAC(AL::Matrix{<:AbstractTensorMap}, C::Matrix{<:AbstractTensorMap})
    return AL .* C 
end

"""
    FLm = FLmap(FLi::Vector{<:AbstractTensorMap}, 
                ALui::Vector{<:AbstractTensorMap},
                ALdir::Vector{<:AbstractTensorMap}, 
                Ati::Vector{<:AbstractTensorMap}, 
                Abi::Vector{<:AbstractTensorMap})

```
  ┌──       ┌──  ALuᵢⱼ ── 
  │         │     │        
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Oᵢⱼ  ── 
  │         │     │        
  └──       └──  ALdᵢᵣⱼ ─ 
```
"""
function FLmap(FLi::Vector{<:AbstractTensorMap}, 
               ALui::Vector{<:AbstractTensorMap},
               ALdir::Vector{<:AbstractTensorMap}, 
               Ati::Vector{<:AbstractTensorMap}, 
               Abi::Vector{<:AbstractTensorMap})
    FLm = [@tensoropt FL[-1 -2 -3; -4] := FL[6 5 4; 1] * ALu[1 2 3; -4] * At[9; 2 -2 8 5] * 
    Ab[3 -3 7 4; 9] * ALd[-1; 6 8 7] for (FL, ALu, ALd, At, Ab) in zip(FLi, ALui, ALdir, Ati, Abi)]

    return circshift(FLm, 1)
end

"""
    FRm = FRmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                Ati::Vector{<:AbstractTensorMap}, 
                Abi::Vector{<:AbstractTensorMap})

```
    ── ARuᵢⱼ  ──┐          ──┐     
        │       │            │     
    ── Oᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁ 
        │       │            │     
    ── ARdᵢᵣⱼ ──┘          ──┘     
```
"""
function FRmap(FRi::Vector{<:AbstractTensorMap}, 
               ARui::Vector{<:AbstractTensorMap}, 
               ARdir::Vector{<:AbstractTensorMap}, 
               Ati::Vector{<:AbstractTensorMap}, 
               Abi::Vector{<:AbstractTensorMap})
    FRm = [@tensoropt FR[-1 -2 -3; -4] := ARu[-1 1 2; 3] * FR[3 4 5; 8] * At[9; 1 4 7 -2] * 
    Ab[2 5 6 -3; 9] * ARd[8; -4 7 6] for (FR, ARu, ARd, At, Ab) in zip(FRi, ARui, ARdir, Ati, Abi)]

    return circshift(FRm, -1)
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                )

```
    ── ARuᵢⱼ  ──┐          ──┐    
        │       Rᵢⱼ  =    ── Rᵢⱼ₋₁  
    ── ARdᵢᵣⱼ ──┘          ──┘     
```
"""
function Rmap(Ri::Vector{<:AbstractTensorMap}, 
              ARui::Vector{<:AbstractTensorMap}, 
              ARdir::Vector{<:AbstractTensorMap})
    Rm = [@tensoropt R[-1; -5] := ARu[-1 2 3; 4] * R[4; 6] * ARd[6; -5 2 3] for (R, ARu, ARd) in zip(Ri, ARui, ARdir)]

    return circshift(Rm, -1)
end

"""
    ACm = ACmap(ACj::Vector{<:AbstractTensorMap}, 
                FLj::Vector{<:AbstractTensorMap}, 
                FRj::Vector{<:AbstractTensorMap},
                Atj::Vector{<:AbstractTensorMap},
                Abj::Vector{<:AbstractTensorMap})

```
                                ┌─────── ACᵢⱼ ─────┐       
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │      
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ   
                                │        │         │      
                                                                
```
"""
function ACmap(ACj::Vector{<:AbstractTensorMap}, 
               FLj::Vector{<:AbstractTensorMap}, 
               FRj::Vector{<:AbstractTensorMap},
               Atj::Vector{<:AbstractTensorMap},
               Abj::Vector{<:AbstractTensorMap})
    ACm = [@tensoropt AC[-1 -2 -3; -4] := AC[1 2 3; 4] * FL[-1 6 5; 1]* At[9; 2 7 -2 6] * 
    Ab[3 8 -3 5; 9] * FR[4 7 8; -4] for (AC, FL, FR, At, Ab) in zip(ACj, FLj, FRj, Atj, Abj)]
    
    return circshift(ACm, 1)
end

"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐       
┌── Cᵢ₊₁ⱼ ──┐       │           │       
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ     
                    │           │       
                                                                       
```
"""
function Cmap(Cj::Vector{<:AbstractTensorMap},
              FLjr::Vector{<:AbstractTensorMap}, 
              FRj::Vector{<:AbstractTensorMap})
    Cm = [@tensoropt C[-1; -2] := C[1; 2] * FL[-1 3 4; 1] * FR[2 3 4; -2] for (C, FL, FR) in zip(Cj, FLjr, FRj)]

    return circshift(Cm, 1)
end