"""List alternating edge factors clockwise, starting at a specified outer bond."""
function _split_boundary_chain(env::SplitCTMRGEnv, (dir, r, c))
    dr, dc = dir == EAST ? (1, 0) : dir == WEST ? (-1, 0) : dir == NORTH ? (0, 1) : (0, -1)
    count = iszero(dr) ? size(env, 3) : size(env, 2)
    firstlayer, secondlayer = env.ket_first[dir] ? (:ket, :bra) : (:bra, :ket)
    return [edge(env, layer, dir, r + i * dr, c + i * dc) for i in 0:(count - 1) for layer in (firstlayer, secondlayer)]
end

"""Determine outer and interlayer basis rotations relative to the previous split environment."""
function compute_relative_phases(final::SplitCTMRGEnv, previous::SplitCTMRGEnv, ::ScramblingEnvGauge)
    gauges = map(eachcoordinate(final, 1:4)) do coordinate
        current = _split_boundary_chain(final, coordinate)
        old = _split_boundary_chain(previous, coordinate)
        outer = _scrambling_gauge(current, old)
        interlayer = _scrambling_gauge(circshift(current, -1), circshift(old, -1))
        return outer, interlayer
    end
    return (; outer = map(first, gauges), interlayer = map(last, gauges))
end

"""Apply the outer gauges to corners and paired outer and interlayer gauges to edge factors."""
function fix_relative_phases(env::SplitCTMRGEnv, signs)
    corner_fns = (fix_gauge_northwest_corner, fix_gauge_northeast_corner, fix_gauge_southeast_corner, fix_gauge_southwest_corner)
    corners = dtmap(eachcoordinate(env, 1:4)) do (dir, r, c)
        return corner_fns[dir]((r, c), env, signs.outer)
    end
    factors = dtmap(eachcoordinate(env, 1:4)) do (dir, r, c)
        dr, dc = dir == EAST ? (1, 0) : dir == WEST ? (-1, 0) : dir == NORTH ? (0, 1) : (0, -1)
        next = (dir, mod1(r + dr, size(env, 2)), mod1(c + dc, size(env, 3)))
        outer, interlayer = signs.outer[dir, r, c], signs.interlayer[dir, r, c]
        outer_next = signs.outer[next...]
        K, B = edge(env, :ket, dir, r, c), edge(env, :bra, dir, r, c)
        return env.ket_first[dir] ?
            (fix_gauge_edge(K, outer, interlayer), fix_gauge_edge(B, interlayer, outer_next)) :
            (fix_gauge_edge(K, interlayer, outer_next), fix_gauge_edge(B, outer, interlayer))
    end
    return SplitCTMRGEnv(corners, map(first, factors), map(last, factors), env.ket_first)
end

"""Determine overlap phases for split corners and individual ket and bra edge factors."""
function compute_global_phases(final::SplitCTMRGEnv, previous::SplitCTMRGEnv)
    return _split_global_phase.(previous.corners, final.corners),
        (; ket = _split_global_phase.(previous.edges_ket, final.edges_ket), bra = _split_global_phase.(previous.edges_bra, final.edges_bra))
end

"""Extract an overlap phase without changing the norm of a split factor."""
function _split_global_phase(previous, current)
    overlap = dot(previous, current)
    return iszero(overlap) ? one(overlap) : overlap / abs(overlap)
end

"""Remove global phases from split environment tensors without changing their norms."""
function fix_global_phases(env::SplitCTMRGEnv, corner_phases, edge_phases)
    return SplitCTMRGEnv(
        env.corners .* inv.(corner_phases),
        env.edges_ket .* inv.(edge_phases.ket), env.edges_bra .* inv.(edge_phases.bra), env.ket_first
    )
end

"""Check split environment compatibility and determine relative gauges and global phases."""
function compute_gauge_fix_gauge(final::SplitCTMRGEnv, previous::SplitCTMRGEnv, alg::ScramblingEnvGauge)
    final.ket_first == previous.ket_first || throw(ArgumentError("Split-edge orientations must match."))
    for field in (:corners, :edges_ket, :edges_bra)
        space.(getproperty(final, field)) == space.(getproperty(previous, field)) ||
            throw(ArgumentError("Split environment spaces must match for gauge fixing."))
    end
    signs = compute_relative_phases(final, previous, alg)
    corner_phases, edge_phases = compute_global_phases(fix_relative_phases(final, signs), previous)
    return signs, corner_phases, edge_phases
end

"""Apply relative gauge rotations and global phase corrections to a split environment."""
function fix_phases(env::SplitCTMRGEnv, signs, corner_phases, edge_phases)
    return fix_global_phases(fix_relative_phases(env, signs), corner_phases, edge_phases)
end

"""Align a split environment with the outer and interlayer gauges of the previous environment."""
function gauge_fix(final::SplitCTMRGEnv, previous::SplitCTMRGEnv, alg::ScramblingEnvGauge)
    return fix_phases(final, compute_gauge_fix_gauge(final, previous, alg)...)
end
