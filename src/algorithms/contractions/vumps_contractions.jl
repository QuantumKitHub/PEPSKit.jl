
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
    ```
    ┌── ALuᵢⱼ  ──      ┌──  
    Lᵢⱼ   |        =   Lᵢⱼ₊₁
    └── ALdᵢᵣⱼ ──      └──  
    ```
"""
function Lmap(Li::Vector{<:AbstractTensorMap}, 
              ALui::Vector{<:AbstractTensorMap}, 
              ALdir::Vector{<:AbstractTensorMap})
    Lm = [@tensoropt L[-6; -4] := ALu[1 2 3; -4] * L[5; 1] * ALd[-6; 5 2 3] for (L, ALu, ALd) in zip(Li, ALui, ALdir)]

    return circshift(Lm, 1)
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
        │       Rᵢⱼ  =       Rᵢⱼ₋₁  
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

function nearest_neighbour_energy(ipeps::InfinitePEPS, Hh, Hv, env::VUMPSEnv)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(ipeps)

    energy_tol = 0
    for j in 1:Nj, i in 1:Ni
        # horizontal contraction
        id = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        @tensoropt oph[-1 -2; -3 -4] := FLo[i,j][18 12 15; 5] * ACu[i,j][5 6 7; 8] * ipeps.A[i,j][-1; 6 13 19 12] * 
                                        conj(ipeps.A[i,j][-3; 7 16 20 15]) * conj(ACd[id,j][18 19 20; 21]) * 
                                        ARu[i,jr][8 9 10; 11] * ipeps.A[i,jr][-2; 9 14 22 13] * 
                                        conj(ipeps.A[i,jr][-4; 10 17 23 16]) * conj(ARd[id,jr][21 22 23; 24]) * FRo[i,jr][11 14 17; 24]

        @tensor eh = oph[1 2; 3 4] * Hh[3 4; 1 2]
        @tensor nh = oph[1 2; 1 2]
        energy_tol += eh / nh
        @show eh / nh

        # vertical contraction
        ir = mod1(i + 1, Ni)
        @tensoropt opv[-1 -2; -3 -4] := FLu[i,j][21 19 20; 18] * ACu[i,j][18 12 15 5] * ipeps.A[i,j][-1; 12 6 13 19] * 
                                    conj(ipeps.A[i,j][-3; 15 7 16 20]) * FRu[i,j][5 6 7; 8] * FLo[ir,j][24 22 23; 21] * 
                                    ipeps.A[ir,j][-2; 13 9 14 22] * conj(ipeps.A[ir,j][-4; 16 10 17 23]) * 
                                    FRo[ir,j][8 9 10; 11] * conj(ACd[id,j][24 14 17; 11])

        @tensor ev = opv[1 2; 3 4] * Hv[3 4; 1 2]
        @tensor nv = opv[1 2; 1 2]
        energy_tol += ev / nv 
        @show ev / nv

        # penalty term 
        energy_tol += 0.1 * abs(eh / nh - eh / nh)
    end

    return energy_tol
end