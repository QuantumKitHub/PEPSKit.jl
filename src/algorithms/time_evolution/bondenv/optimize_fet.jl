"""
Full environment truncation (FET) of the bond between `a` and `b`.
```
    |------env------|
    |→ a† ===== b† ←|
    |   ↑       ↑   |
    |←- a ===== b -→|
    |---------------|
```

Reference: Physical Review B 98, 085155 (2018)
"""
function bond_optimize(
    env::BondEnv{T,S},
    a::AbstractTensor{T,S,3},
    b::AbstractTensor{T,S,3},
    alg::FullEnvTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    @assert [isdual(space(a, ax)) for ax in 1:2] == [0, 0]
    @assert [isdual(space(b, ax)) for ax in 2:3] == [0, 0]
    #= initialize bond matrix using QR as `Ra Lb`

            ↑    ↑               ↑               ↑
        ←-- a == b --→   ==>   ← Qa ← Ra == Lb → Qb →
    =#
    Qa, Ra = leftorth(a, ((1, 2), (3,)))
    Lb, Qb = rightorth(b, ((1,), (2, 3)))
    flipper = isomorphism(flip(space(Qb, 1)), space(Qb, 1))
    Lb = Lb * flipper'
    Qb = twist(flipper * Qb, 1)
    @tensor b0[-1 -2] := Ra[-1 1] * Lb[1 -2]
    #= initialize bond environment around `Ra Lb`

        |-------env-------|
        |→ Qa†→     ← Qb†←|
        |  ↑          ↑   |
        |← Qa ←     → Qb →|
        |-----------------|
    =#
    @tensor env2[-1 -2; -3 -4] := (
        env[1 2; 3 4] * conj(Qa[1 5 -1]) * conj(Qb[-2 6 2]) * Qa[3 5 -3] * Qb[-4 6 4]
    )
    # optimize bond matrix
    u, s, vh, info = fullenv_truncate(env2, b0, alg; flip_s=false)
    s /= norm(s, Inf)
    # truncate a, b tensors with u, s, vh
    @tensor a[-1 -2; -3] := Qa[-1 -2 3] * u[3 -3]
    @tensor b[-1; -2 -3] := vh[-1 1] * Qb[1 -2 -3]
    return a, s, b, info
end
