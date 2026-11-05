function gauge_fix(envfinal::CTMRGEnvTriangular, envprev::CTMRGEnvTriangular, alg::ScramblingEnvGauge)
    signs, corner_phases, edge_phases = compute_gauge_fix_gauge(envfinal, envprev, alg)
    return fix_phases(envfinal, signs, corner_phases, edge_phases)
end

function compute_gauge_fix_gauge(envfinal::CTMRGEnvTriangular, envprev::CTMRGEnvTriangular, alg::ScramblingEnvGauge)
    same_spaces = all(zip((envfinal.C, envfinal.Ea, envfinal.Eb), (envprev.C, envprev.Ea, envprev.Eb))) do (final, prev)
        return size(final) == size(prev) && all(space.(final) .== space.(prev))
    end
    same_spaces || throw(ArgumentError("Spaces of envfinal and envprev are not the same"))
    signs = compute_relative_phases(envfinal, envprev, alg)
    corner_phases, edge_phases = compute_global_phases(fix_relative_phases(envfinal, signs), envprev)
    return signs, corner_phases, edge_phases
end

# Follow the boundary tangent, alternating the two edge halves at each site.
function _triangular_edge_chain(env::CTMRGEnvTriangular, dir, r, c)
    tensors = [env.Eb[dir, r, c], env.Ea[dir, r, c]]
    unitcell = size(env)[2:3]
    _, r′, c′ = _coordinates(dir, 2, r, c, unitcell)
    while (r′, c′) != (r, c)
        push!(tensors, env.Eb[dir, r′, c′], env.Ea[dir, r′, c′])
        _, r′, c′ = _coordinates(dir, 2, r′, c′, unitcell)
    end
    return tensors
end

function compute_relative_phases(envfinal::CTMRGEnvTriangular, envprev::CTMRGEnvTriangular, ::ScramblingEnvGauge)
    signs = map((1, 0)) do shift
        return map(eachcoordinate(envfinal, 1:6)) do (dir, r, c)
            Tsprev = circshift(_triangular_edge_chain(envprev, dir, r, c), -shift)
            Tsfinal = circshift(_triangular_edge_chain(envfinal, dir, r, c), -shift)
            M = map(t -> randn(storagetype(t), space(t)), Tsfinal)
            ρprev = right_transfermatrix_fixedpoint(Tsprev, M, Arnoldi())
            ρfinal = right_transfermatrix_fixedpoint(Tsfinal, M, Arnoldi())
            Qprev, = left_orth!(ρprev; positive = true)
            Qfinal, = left_orth!(ρfinal; positive = true)
            return Qprev * Qfinal'
        end
    end
    return (; Ea = signs[1], Eb = signs[2])
end

function fix_relative_phases(env::CTMRGEnvTriangular, signs)
    unitcell = size(env)[2:3]
    # Triangular corners have the same tensor layout as the edge halves.
    C = map(eachcoordinate(env, 1:6)) do (dir, r, c)
        next = _coordinates(dir, 2, r, c, unitcell)
        return fix_gauge_edge(env.C[dir, r, c], signs.Eb[mod1(dir - 1, 6), r, c], signs.Eb[next...])
    end
    Ea = map(eachcoordinate(env, 1:6)) do (dir, r, c)
        next = _coordinates(dir, 2, r, c, unitcell)
        return fix_gauge_edge(env.Ea[dir, r, c], signs.Ea[dir, r, c], signs.Eb[next...])
    end
    Eb = map(eachcoordinate(env, 1:6)) do (dir, r, c)
        return fix_gauge_edge(env.Eb[dir, r, c], signs.Eb[dir, r, c], signs.Ea[dir, r, c])
    end
    return CTMRGEnvTriangular(C, Ea, Eb)
end

function compute_global_phases(envfix::CTMRGEnvTriangular, envprev::CTMRGEnvTriangular)
    corner_phases = dot.(envprev.C, envfix.C)
    edge_phases = (; Ea = dot.(envprev.Ea, envfix.Ea), Eb = dot.(envprev.Eb, envfix.Eb))
    return corner_phases, edge_phases
end

function fix_global_phases(env::CTMRGEnvTriangular, corner_phases, edge_phases)
    return CTMRGEnvTriangular(env.C .* inv.(corner_phases), env.Ea .* inv.(edge_phases.Ea), env.Eb .* inv.(edge_phases.Eb))
end

function fix_phases(env::CTMRGEnvTriangular, signs, corner_phases, edge_phases)
    return fix_global_phases(fix_relative_phases(env, signs), corner_phases, edge_phases)
end

function calc_elementwise_convergence(envfinal::CTMRGEnvTriangular, envfix::CTMRGEnvTriangular; atol::Real = 1.0e-6)
    return max(norm(envfinal.C .- envfix.C, Inf), norm(envfinal.Ea .- envfix.Ea, Inf), norm(envfinal.Eb .- envfix.Eb, Inf))
end
