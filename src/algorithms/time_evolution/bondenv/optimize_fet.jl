"""
    bond_optimize(benv::BondEnv{T,S}, a::AbstractTensor{T,S,3}, b::AbstractTensor{T,S,3}, alg) where {T<:Number,S<:ElementarySpace}

Truncation of the bond between `a` and `b`.
```
    ┌-----------------------┐
    |   ┌----┐              |
    └---|    |-- a† == b† --┘
        |benv|   ↑     ↑   
    ┌---|    |-- a === b ---┐
    |   └----┘              |
    └-----------------------┘
```
The truncation algorithm `alg` can be either `FullEnvTruncation` or `ALSTruncation`. 
The index order of `a` or `b` is
```
        2
        |
    1 -a/b- 3
```
"""
function bond_optimize(
    benv::BondEnv{T,S},
    a::AbstractTensor{T,S,3},
    b::AbstractTensor{T,S,3},
    alg::FullEnvTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    @assert codomain(benv) == domain(benv)
    #= initialize bond matrix using QR as `Ra Lb`

            ↑    ↑               ↑               ↑
        --- a == b ---   ==>   - Qa - Ra == Rb - Qb -
    =#
    Qa, Ra = leftorth(a, ((1, 2), (3,)))
    Qb, Rb = leftorth(b, ((2, 3), (1,)))
    isdual(codomain(Ra, 1)) && twist!(Ra, 1)
    isdual(codomain(Rb, 1)) && twist!(Rb, 1)
    @tensor b0[-1 -2] := Ra[-1 1] * Rb[-2 1]
    #= initialize bond environment around `Ra Lb`

        ┌--------------------------------------┐
        |   ┌----┐                             |
        └---|    |- 1 - Qa†- -1   -2 - Qb†- 2 -┘
            |    |      ↑              ↑
            |benv|      5              6
            |    |      ↑              ↑
        ┌---|    |- 3 - Qa - -3   -4 - Qb - 4 -┐
        |   └----┘                             |
        └--------------------------------------┘
    =#
    @tensor benv2[-1 -2; -3 -4] := (
        benv[1 2; 3 4] * conj(Qa[1 5 -1]) * conj(Qb[6 2 -2]) * Qa[3 5 -3] * Qb[6 4 -4]
    )
    # optimize bond matrix
    u, s, vh, info = fullenv_truncate(benv2, b0, alg)
    # truncate a, b tensors with u, s, vh
    @tensor a[-1 -2; -3] := Qa[-1 -2 3] * u[3 -3]
    @tensor b[-1; -2 -3] := vh[-1 1] * Qb[-2 -3 1]
    return a, s, b, info
end
