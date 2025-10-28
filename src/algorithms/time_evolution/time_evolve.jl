abstract type TimeEvolution end

function MPSKit.time_evolve(alg::Alg) where {Alg <: TimeEvolution}
    time_start = time()
    result = nothing
    for state in alg
        result = state
    end
    time_end = time()
    @info @sprintf("Simple update finished. Total time elasped: %.2f s", time_end - time_start)
    return result
end
