"""Represent a split boundary as a doubled MPS chain of alternating ket and bra factors, starting at an outer bond."""
function _boundary_chain(env::SplitCTMRGEnv, coordinate)
    dir = first(coordinate)
    firstlayer, secondlayer = env.ket_first[dir] ? (:ket, :bra) : (:bra, :ket)
    return [edge(env, layer, I...) for I in _boundary_coordinates(env, coordinate) for layer in (firstlayer, secondlayer)]
end

"""Determine outer and interlayer basis rotations relative to the previous split environment."""
function compute_relative_phases(final::SplitCTMRGEnv, previous::SplitCTMRGEnv, ::ScramblingEnvGauge)
    gauges = map(eachcoordinate(final, 1:4)) do coordinate
        current = _boundary_chain(final, coordinate)
        old = _boundary_chain(previous, coordinate)
        outer = _scrambling_gauge(current, old)
        # The same method on the doubled chain, shifted to the interlayer cut.
        interlayer = _scrambling_gauge(circshift(current, -1), circshift(old, -1))
        return outer, interlayer
    end
    return (; outer = map(first, gauges), interlayer = map(last, gauges))
end

"""Apply the outer gauges to corners and paired outer and interlayer gauges to edge factors."""
function fix_relative_phases(env::SplitCTMRGEnv, signs)
    corners = _fix_gauge_corners(env, signs.outer)
    factors = dtmap(eachcoordinate(env, 1:4)) do (dir, r, c)
        dr, dc = _boundary_step(dir)
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

"""Check that split environments have matching orientations and spaces for gauge alignment."""
function _check_gauge_compatibility(final::SplitCTMRGEnv, previous::SplitCTMRGEnv)
    final.ket_first == previous.ket_first || throw(ArgumentError("Split-edge orientations must match."))
    for field in (:corners, :edges_ket, :edges_bra)
        space.(getproperty(final, field)) == space.(getproperty(previous, field)) ||
            throw(ArgumentError("Split environment spaces must match for gauge fixing."))
    end
    return nothing
end
