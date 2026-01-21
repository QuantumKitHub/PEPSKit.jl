"""
    gauge_transform(env::CTMRGEnv, XXinv)

Transform the CTMRG environment `env` of an InfinitePEPS after it is
transformed by gauge transformations `XXinv` on its virtual bonds.

`XXinv` consists of `(X, X⁻¹)` pairs on each virtual bond of the InfinitePEPS.
```
    T[r-1,c]
    |
    X⁻¹
    |   [NORTH,r,c]
    X
    |
    T[r,c]----X---X⁻¹----T[r,c+1]
            [EAST,r,c]
```
"""
function gauge_transform(env::CTMRGEnv, XXinv)
    edges = map(eachcoordinate(env, 1:4)) do (d, r, c)
        if d == NORTH
            X⁻¹ = XXinv[NORTH, _next(r, end), c][2]
            return @tensor edge[χ1 d0 d1; χ2] :=
                env.edges[d, r, c][χ1 d0′ d1′; χ2] * X⁻¹[d0; d0′] * conj(X⁻¹[d1; d1′])
        elseif d == EAST
            X⁻¹ = XXinv[EAST, r, _prev(c, end)][2]
            return @tensor edge[χ1 d0 d1; χ2] :=
                env.edges[d, r, c][χ1 d0′ d1′; χ2] * X⁻¹[d0; d0′] * conj(X⁻¹[d1; d1′])
        elseif d == SOUTH
            X = XXinv[NORTH, r, c][1]
            return @tensor edge[χ1 d0 d1; χ2] :=
                env.edges[d, r, c][χ1 d0′ d1′; χ2] * X[d0′; d0] * conj(X[d1′; d1])
        else # d == WEST
            X = XXinv[EAST, r, c][1]
            return @tensor edge[χ1 d0 d1; χ2] :=
                env.edges[d, r, c][χ1 d0′ d1′; χ2] * X[d0′; d0] * conj(X[d1′; d1])
        end
    end
    # corners are unaffected
    return CTMRGEnv(env.corners, edges)
end
