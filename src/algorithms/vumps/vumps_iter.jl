function vumps_iter(ψ::InfiniteMPS, H, alg, envs, ϵ)
    (st, pr, de) = vumps_iter(convert(MPSMultiline, ψ), Multiline([H]), alg, envs, ϵ)
    return convert(InfiniteMPS, st), pr, de
end

using MPSKit: gaugefix!, _firstspace
function vumps_iter(ψ::MPSMultiline, H, alg::VUMPS, envs, ϵ)
    ACs = Zygote.Buffer(ψ.AC)
    Cs = Zygote.Buffer(ψ.CR)

    iter = alg.maxiter
    alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
    for col in 1:size(ψ, 2)
        H_AC = ∂∂AC(col, ψ, H, envs)
        ac = RecursiveVec(ψ.AC[:, col])
        _, ac′ = MPSKit.fixedpoint(H_AC, ac, :LM, alg_eigsolve)
        ACs[:, col] = ac′.vecs[:]

        H_C = ∂∂C(col, ψ, H, envs)
        c = RecursiveVec(ψ.CR[:, col])
        _, c′ = MPSKit.fixedpoint(H_C, c, :LM, alg_eigsolve)
        Cs[:, col] = c′.vecs[:]
    end

    # normalize
    ACs = [AC / norm(AC) for AC in copy(ACs)]
    Cs = [C / norm(C) for C in copy(Cs)]

    ALs = [regauge(AC, C) for (AC, C) in zip(ACs, Cs)]
    ARs = [regauge(C, AC) for (C, AC) in zip(Cs, ACs)]
    ACs = [AL * C for (AL, C) in zip(ALs, Cs)]
    # ACs2 = [_transpose_front(C * _transpose_tail(AR)) for (C, AR) in zip(Cs, ARs)]
    # @show norm(ACs1[1] - ACs2[1])

    # ψ = MPSMultiline(vec([InfiniteMPS(ALs[i, :], ARs[i, :], Cs[i, :], ACs[i, :]) for i in size(ALs, 1)]))

    data = map(eachrow(ALs), eachrow(ARs), eachrow(Cs), eachrow(ACs)) do ALrow, ARrow, Crow, ACrow
        InfiniteMPS(ALrow, ARrow, Crow, ACrow)
    end
    ψ = MPSMultiline(data)

    # alg_environments = updatetol(alg.alg_environments, iter, ϵ)
    # @show typeof(envs) typeof(ψ)
    # recalculate!(envs, ψ; alg_environments.tol)

    return ψ, envs, ϵ
end

function regauge(AC::GenericMPSTensor, CR::MPSBondTensor; alg=QRpos())
    Q_AC, _ = leftorth(AC; alg)
    Q_C, _ = leftorth(CR; alg)
    return Q_AC * Q_C'
end

function regauge(CL::MPSBondTensor, AC::GenericMPSTensor; alg=LQpos())
    AC_tail = _transpose_tail(AC)
    _, Q_AC = rightorth(AC_tail; alg)
    _, Q_C = rightorth(CL; alg)
    AR_tail = Q_C' * Q_AC
    return _transpose_front(AR_tail)
end

