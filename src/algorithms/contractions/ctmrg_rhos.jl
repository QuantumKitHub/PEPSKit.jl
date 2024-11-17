"""
Calculate 1-site rho at site `(r,c)`
```
    C1 - χ4 - T1 - χ6 - C2  r-1
    |         ‖         |
    χ2        DN        χ8
    |         ‖         |
    T4 = DW =k/b = DE = T2  r
    |         ‖         |
    χ1        DS        χ7
    |         ‖         |
    C4 - χ3 - T3 - χ5 - C3  r+1
    c-1       c        c+1
```
Indices d0, d1 are physical indices of ket, bra
"""
function calrho_site(
    row::Int, col::Int, envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    N1, N2 = size(ket)
    @assert 1 <= row <= N1 && 1 <= col <= N2
    rp1, rm1 = _next(row, N1), _prev(row, N1)
    cp1, cm1 = _next(col, N2), _prev(col, N2)
    tket, tbra = ket[row, col], bra[row, col]
    c1 = envs.corners[1, rm1, cm1]
    t1 = envs.edges[1, rm1, col]
    c2 = envs.corners[2, rm1, cp1]
    t2 = envs.edges[2, row, cp1]
    c3 = envs.corners[3, rp1, cp1]
    t3 = envs.edges[3, rp1, col]
    c4 = envs.corners[4, rp1, cm1]
    t4 = envs.edges[4, row, cm1]
    PEPSKit.@autoopt @tensor rho1[d1; d0] := (
        c4[χ3, χ1] *
        t4[χ1, DW0, DW1, χ2] *
        c1[χ2, χ4] *
        t3[χ5, DS0, DS1, χ3] *
        tket[d0, DN0, DE0, DS0, DW0] *
        conj(tbra[d1, DN1, DE1, DS1, DW1]) *
        t1[χ4, DN0, DN1, χ6] *
        c3[χ7, χ5] *
        t2[χ8, DE0, DE1, χ7] *
        c2[χ6, χ8]
    )
    return rho1
end

"""
Calculate 2-site rho on sites `(r,c)(r,c+1)`
```
    C1 - χ4 - T1 - χ6 - T1 - χ8 - C2    r-1
    |         ‖         ‖         |
    χ2       DN1       DN2        χ10
    |         ‖         ‖         |
    T4 = DW =k/b = DM =k/b = DE = T2    r
    |         ‖         ‖         |
    χ1       DS1       DS2        χ9
    |         ‖         ‖         |
    C4 - χ3 - T3 - χ5 - T3 - χ7 - C3    r+1
    c-1       c        c+1       c+2
```
Indices d0, d1 are physical indices of ket, bra
"""
function calrho_bondx(
    row::Int, col::Int, envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    N1, N2 = size(ket)
    @assert 1 <= row <= N1 && 1 <= col <= N2
    rp1, rm1 = _next(row, N1), _prev(row, N1)
    cp1, cm1 = _next(col, N2), _prev(col, N2)
    cp2 = _next(cp1, N2)
    tket1, tbra1 = ket[row, col], bra[row, col]
    tket2, tbra2 = ket[row, cp1], bra[row, cp1]
    c1 = envs.corners[1, rm1, cm1]
    t11, t12 = envs.edges[1, rm1, col], envs.edges[1, rm1, cp1]
    c2 = envs.corners[2, rm1, cp2]
    t2 = envs.edges[2, row, cp2]
    c3 = envs.corners[3, rp1, cp2]
    t31, t32 = envs.edges[3, rp1, col], envs.edges[3, rp1, cp1]
    c4 = envs.corners[4, rp1, cm1]
    t4 = envs.edges[4, row, cm1]
    PEPSKit.@autoopt @tensor rho2[d11, d21; d10, d20] := (
        c4[χ3, χ1] *
        t4[χ1, DW0, DW1, χ2] *
        c1[χ2, χ4] *
        t31[χ5, DS10, DS11, χ3] *
        tket1[d10, DN10, DM0, DS10, DW0] *
        conj(tbra1[d11, DN11, DM1, DS11, DW1]) *
        t11[χ4, DN10, DN11, χ6] *
        t32[χ7, DS20, DS21, χ5] *
        tket2[d20, DN20, DE0, DS20, DM0] *
        conj(tbra2[d21, DN21, DE1, DS21, DM1]) *
        t12[χ6, DN20, DN21, χ8] *
        c3[χ9, χ7] *
        t2[χ10, DE0, DE1, χ9] *
        c2[χ8, χ10]
    )
    return rho2
end

"""
Calculate 2-site rho on sites `(r,c)(r-1,c)`
```
    C1 - χ9 - T1 -χ10 - C2  r-2
    |         ‖         |
    χ7        DN        χ8
    |         ‖         |
    T4 = DW2=k/b =DE2 = T2  r-1
    |         ‖         |
    χ5        DM        χ6
    |         ‖         |
    T4 = DW1=k/b =DE1 = T2  r
    |         ‖         |
    χ3        DS        χ4
    |         ‖         |
    C4 - χ1 - T3 - χ2 - C3  r+1
    c-1       c        c+1
```
Indices d0, d1 are physical indices of ket, bra
"""
function calrho_bondy(
    row::Int, col::Int, envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    N1, N2 = size(ket)
    @assert 1 <= row <= N1 && 1 <= col <= N2
    rp1, rm1 = _next(row, N1), _prev(row, N1)
    cp1, cm1 = _next(col, N2), _prev(col, N2)
    rm2 = _prev(rm1, N1)
    tket1, tbra1 = ket[row, col], bra[row, col]
    tket2, tbra2 = ket[rm1, col], bra[rm1, col]
    c1 = envs.corners[1, rm2, cm1]
    t1 = envs.edges[1, rm2, col]
    c2 = envs.corners[2, rm2, cp1]
    t21, t22 = envs.edges[2, row, cp1], envs.edges[2, rm1, cp1]
    c3 = envs.corners[3, rp1, cp1]
    t3 = envs.edges[3, rp1, col]
    c4 = envs.corners[4, rp1, cm1]
    t41, t42 = envs.edges[4, row, cm1], envs.edges[4, rm1, cm1]
    PEPSKit.@autoopt @tensor rho2[d11, d21; d10, d20] := (
        c4[χ1, χ3] *
        t3[χ2, DS0, DS1, χ1] *
        c3[χ4, χ2] *
        t41[χ3, DW10, DW11, χ5] *
        tket1[d10, DM0, DE10, DS0, DW10] *
        conj(tbra1[d11, DM1, DE11, DS1, DW11]) *
        t21[χ6, DE10, DE11, χ4] *
        t42[χ5, DW20, DW21, χ7] *
        tket2[d20, DN0, DE20, DM0, DW20] *
        conj(tbra2[d21, DN1, DE21, DM1, DW21]) *
        t22[χ8, DE20, DE21, χ6] *
        c1[χ7, χ9] *
        t1[χ9, DN0, DN1, χ10] *
        c2[χ10, χ8]
    )
    return rho2
end

"""
Calculate rho for all sites
"""
function calrho_allsites(envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket)
    Nr, Nc = size(ket)
    return collect(
        calrho_site(r, c, envs, ket, bra) for (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
end

"""
Calculate rho for all nearest-neighbor bonds
"""
function calrho_allnbs(envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket)
    Nr, Nc = size(ket)
    rhoxss = collect(
        calrho_bondx(r, c, envs, ket, bra) for (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    rhoyss = collect(
        calrho_bondy(r, c, envs, ket, bra) for (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    return [rhoxss, rhoyss]
end

"""
Calculate rho for all sites and nearest-neighbor bonds
"""
function calrho_all(envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket)
    rho1ss = calrho_allsites(envs, ket, bra)
    rho2sss = calrho_allnbs(envs, ket, bra)
    return rho1ss, rho2sss
end
