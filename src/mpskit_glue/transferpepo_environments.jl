
function MPSKit.environments(state::InfiniteMPS, O::InfiniteTransferPEPO; kwargs...)
    return environments(
        convert(MPSMultiline, state), convert(TransferPEPOMultiline, O); kwargs...
    )
end

function MPSKit.environments(
    state::MPSMultiline, O::TransferPEPOMultiline; solver=MPSKit.Defaults.eigsolver
)
    (lw, rw) = MPSKit.mixed_fixpoints(state, O, state; solver)
    return MPSKit.PerMPOInfEnv(nothing, O, state, solver, lw, rw, ReentrantLock())
end

function MPSKit.mixed_fixpoints(
    above::MPSMultiline,
    O::TransferPEPOMultiline,
    below::MPSMultiline,
    init=gen_init_fps(above, O, below);
    solver=MPSKit.Defaults.eigsolver,
)
    T = eltype(above)

    (numrows, numcols) = size(above)
    @assert size(above) == size(O)
    @assert size(below) == size(O)

    envtype = eltype(init[1])
    lefties = PeriodicArray{envtype,2}(undef, numrows, numcols)
    righties = PeriodicArray{envtype,2}(undef, numrows, numcols)

    @threads for cr in 1:numrows
        c_above = above[cr]  # TODO: Update index convention to above[cr - 1]
        c_below = below[cr + 1]

        (L0, R0) = init[cr]

        @sync begin
            Threads.@spawn begin
                E_LL = TransferMatrix($c_above.AL, $O[cr], $c_below.AL)
                (_, Ls, convhist) = eigsolve(flip(E_LL), $L0, 1, :LM, $solver)
                convhist.converged < 1 &&
                    @info "left eigenvalue failed to converge $(convhist.normres)"
                L0 = first(Ls)
            end

            Threads.@spawn begin
                E_RR = TransferMatrix($c_above.AR, $O[cr], $c_below.AR)
                (_, Rs, convhist) = eigsolve(E_RR, $R0, 1, :LM, $solver)
                convhist.converged < 1 &&
                    @info "right eigenvalue failed to converge $(convhist.normres)"
                R0 = first(Rs)
            end
        end

        lefties[cr, 1] = L0
        for loc in 2:numcols
            lefties[cr, loc] =
                lefties[cr, loc - 1] *
                TransferMatrix(c_above.AL[loc - 1], O[cr, loc - 1], c_below.AL[loc - 1])
        end

        renormfact::scalartype(T) = dot(c_below.CR[0], PEPO_∂∂C(L0, R0) * c_above.CR[0])

        righties[cr, end] = R0 / sqrt(renormfact)
        lefties[cr, 1] /= sqrt(renormfact)

        for loc in (numcols - 1):-1:1
            righties[cr, loc] =
                TransferMatrix(c_above.AR[loc + 1], O[cr, loc + 1], c_below.AR[loc + 1]) *
                righties[cr, loc + 1]

            renormfact = dot(
                c_below.CR[loc],
                PEPO_∂∂C(lefties[cr, loc + 1], righties[cr, loc]) * c_above.CR[loc],
            )
            righties[cr, loc] /= sqrt(renormfact)
            lefties[cr, loc + 1] /= sqrt(renormfact)
        end
    end

    return (lefties, righties)
end

function gen_init_fps(above::MPSMultiline, O::TransferPEPOMultiline, below::MPSMultiline)
    T = eltype(above)

    map(1:size(O, 1)) do cr
        L0::T = TensorMap(
            rand,
            scalartype(T),
            left_virtualspace(below, cr + 1, 0) * prod(adjoint.(west_spaces(O[cr], 1))),
            left_virtualspace(above, cr, 0),  # TODO: Update index convention to above[cr - 1]
        )
        R0::T = TensorMap(
            rand,
            scalartype(T),
            right_virtualspace(above, cr, 0) * prod(adjoint.(east_spaces(O[cr], 1))),
            right_virtualspace(below, cr + 1, 0),
        )
        (L0, R0)
    end
end
