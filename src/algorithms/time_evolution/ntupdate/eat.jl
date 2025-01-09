"""
Environment assisted truncation (EAT)

Reference: Physical Review B 106, 195105 (2022)
"""
function ea_truncate(
    env::BondEnv, Ra::AbstractTensorMap, Rb::AbstractTensorMap, trscheme::TruncationScheme
)
    #= Decompose environment and eigen-decompose each part

                                    ↑   ↑
        ↑       ↑       ↑   ↑       Ua  Ub
        |--env--|   ≈   Ga  Gb  =   λa  λb
        ↑       ↑       ↑   ↑       Ua† Ub†
                                    ↑   ↑
    =#
    Ga, Gb = absorb_s(tsvd(env, ((1, 3), (2, 4)); trunc=truncdim(1))[1:3]...)
    return error("Not implemented")
end
