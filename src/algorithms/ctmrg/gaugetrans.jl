"""
    gauge_transform(env::CTMRGEnv, gauge::VirtualGaugeTransform)

Transform the network-facing virtual legs of `env` to compensate for a virtual gauge transformation of its state.
A gauge of shape `(2, rows, cols)` acts on a single-layer PEPO environment or a double-layer PEPS/PEPO environment, as determined by the number of virtual legs on each edge.
In a double layer, the same gauge acts on the ket and its complex conjugate on the bra.
A PEPO gauge with an explicit layer dimension is accepted when it contains one layer.
The corner tensors and boundary legs are unchanged.
"""
function gauge_transform(env::CTMRGEnv, gauge::VirtualGaugeTransform{3})
    _check_gauge_size(gauge, size(env)[2:3])
    X, Xinv = gauge.matrices, gauge.inverses
    edges = map(eachcoordinate(env, 1:4)) do (d, r, c)
        factor = if d == NORTH
            Xinv[NORTH, _next(r, size(X, 2)), c]
        elseif d == EAST
            Xinv[EAST, r, _prev(c, size(X, 3))]
        elseif d == SOUTH
            X[NORTH, r, c]
        else # d == WEST
            X[EAST, r, c]
        end
        return _gauge_transform_virtual_edge(env.edges[d, r, c], factor, d == NORTH || d == EAST)
    end
    return CTMRGEnv(env.corners, edges)
end
function gauge_transform(env::CTMRGEnv, gauge::VirtualGaugeTransform{4})
    size(gauge.matrices, 4) == 1 || throw(ArgumentError("Virtual environment transformations require a single-layer gauge"))
    return gauge_transform(
        env, VirtualGaugeTransform(dropdims(gauge.matrices; dims = 4), dropdims(gauge.inverses; dims = 4))
    )
end

gauge_transform(env::CTMRGEnv, pairs::AbstractArray{<:Tuple{MPSBondTensor, MPSBondTensor}, 3}) =
    gauge_transform(env, VirtualGaugeTransform(pairs))

"""Apply a compensating virtual gauge to a single-layer or double-layer edge tensor."""
function _gauge_transform_virtual_edge(edge::CTMRGEdgeTensor, X::MPSBondTensor, incoming::Bool)
    if numout(edge) == 2
        if incoming
            return @tensor t[a p; b] := edge[a q; b] * X[p; q]
        else
            return @tensor t[a p; b] := edge[a q; b] * X[q; p]
        end
    elseif numout(edge) == 3
        if incoming
            return @tensor t[a p q; b] := edge[a p′ q′; b] * X[p; p′] * conj(X[q; q′])
        else
            return @tensor t[a p q; b] := edge[a p′ q′; b] * X[p′; p] * conj(X[q′; q])
        end
    end
    throw(ArgumentError("Virtual environment transformations require one or two network-facing legs per edge"))
end

"""
    gauge_transform(env::CTMRGEnv, gauge::CTMRGEnvGaugeTransform)

Transform the boundary legs of the corners and edges of `env`, leaving its network-facing legs unchanged.
Each boundary bond receives a matrix and its inverse, preserving contracted observables for general invertible gauges.
This also supports environments with more than two network-facing legs per edge.
"""
function gauge_transform(env::CTMRGEnv, gauge::CTMRGEnvGaugeTransform)
    _check_gauge_size(gauge, size(env)[2:3])
    X, Xinv = gauge.matrices, gauge.inverses
    nr, nc = size(env)[2:3]
    corners = map(eachcoordinate(env, 1:4)) do (d, r, c)
        outdir, indir = _prev(d, 4), d
        ri, ci = _environment_gauge_neighbor(indir, r, c, nr, nc)
        return fix_gauge_corner(env.corners[d, r, c], X[outdir, r, c], Xinv[indir, ri, ci]')
    end
    edges = map(eachcoordinate(env, 1:4)) do (d, r, c)
        ri, ci = _environment_gauge_neighbor(d, r, c, nr, nc)
        return fix_gauge_edge(env.edges[d, r, c], X[d, r, c], Xinv[d, ri, ci]')
    end
    return CTMRGEnv(corners, edges)
end

"""Find the next boundary bond along an edge, respecting the periodic unit cell."""
function _environment_gauge_neighbor(d::Int, r::Int, c::Int, nr::Int, nc::Int)
    d == NORTH && return r, _next(c, nc)
    d == EAST && return _next(r, nr), c
    d == SOUTH && return r, _prev(c, nc)
    return _prev(r, nr), c
end
