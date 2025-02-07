"""
    bond_optimize(env::BondEnv{T,S}, a::AbstractTensor{T,S,3}, b::AbstractTensor{T,S,3}, alg) where {T<:Number,S<:ElementarySpace}

Truncation of the bond between `a` and `b`.
```
    |------env------|
    |- a† ===== b† -|
    |   ↑       ↑   |
    |-- a ===== b --|
    |---------------|
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
    env::BondEnv{T,S},
    a::AbstractTensor{T,S,3},
    b::AbstractTensor{T,S,3},
    alg::FullEnvTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    #= initialize bond matrix using QR as `Ra Lb`

            ↑    ↑               ↑               ↑
        --- a == b ---   ==>   - Qa - Ra == Lb - Qb -
    =#
    Qa, Ra = leftorth(a, ((1, 2), (3,)))
    Lb, Qb = rightorth(b, ((1,), (2, 3)))
    @tensor b0[-1 -2] := Ra[-1 1] * Lb[1 -2]
    #= initialize bond environment around `Ra Lb`

        |-------env-------|
        |- Qa†-     - Qb†-|
        |  ↑          ↑   |
        |- Qa -     - Qb -|
        |-----------------|
    =#
    @tensor env2[-1 -2; -3 -4] := (
        env[1 2; 3 4] * conj(Qa[1 5 -1]) * conj(Qb[-2 6 2]) * Qa[3 5 -3] * Qb[-4 6 4]
    )
    # optimize bond matrix
    u, s, vh, info = fullenv_truncate(env2, b0, alg)
    s /= norm(s, Inf)
    # truncate a, b tensors with u, s, vh
    @tensor a[-1 -2; -3] := Qa[-1 -2 3] * u[3 -3]
    @tensor b[-1; -2 -3] := vh[-1 1] * Qb[1 -2 -3]
    return a, s, b, info
end
