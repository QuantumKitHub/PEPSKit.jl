# some basic symmetrization routines for PEPS

abstract type SymmetrizationStyle end

struct None <: SymmetrizationStyle end
struct Depth <: SymmetrizationStyle end
struct Width <: SymmetrizationStyle end
struct Rot <: SymmetrizationStyle end
struct Full <: SymmetrizationStyle end

# some rather shady definitions for 'hermitian conjugate' at the level of a single tensor
function herm_depth(x::PEPSTensor)
    return permute(x', ((5,), (3, 2, 1, 4)))
end
function herm_depth(x::PEPOTensor)
    return permute(x', ((5, 6), (3, 2, 1, 4)))
end

function herm_width(x::PEPSTensor)
    x = Permute(Conj(x), [1, 4, 3, 2, 5:(x.legs)])
    return permute(x', ((5,), (1, 4, 3, 2)))
end
function herm_width(x::PEPOTensor)
    return permute(x', ((5, 6), (1, 4, 3, 2)))
end

function herm_height(x::PEPOTensor)
    x = Permute(Conj(x), [1:4, 6, 5])
    return permute(x', ((6, 5), (1, 2, 3, 4)))
end

# hermitian invariance

# make two TensorMap's have the same spaces, by force if necessary
# this is definitely not what you would want to do, but it circumvents having to think
# about what hermiticity means at the level of transfer operators, which is something
function _make_it_fit(
    y::AbstractTensorMap{S,N₁,N₂}, x::AbstractTensorMap{S,N₁,N₂}
) where {S<:IndexSpace,N₁,N₂}
    for i in 1:(N₁ + N₂)
        if space(x, i) ≠ space(y, i)
            f = unitary(space(x, i) ← space(y, i))
            y = permute(
                ncon([f, y], [[-i, 1], [-(1:(i - 1))..., 1, -((i + 1):(N₁ + N₂))...]]),
                (Tuple(1:N₁), Tuple((N₁ + 1):(N₁ + N₂))),
            )
        end
    end
    return y
end

function herm_depth_inv(x::Union{PEPSTensor,PEPOTensor})
    return 0.5 * (x + _make_it_fit(herm_depth(x), x))
end

function herm_width_inv(x::Union{PEPSTensor,PEPOTensor})
    return 0.5 * (x + _make_it_fit(herm_width(x), x))
end

function herm_height_inv(x::Union{PEPSTensor,PEPOTensor})
    return 0.5 * (x + _make_it_fit(herm_height(x), x))
end

# rotation invariance

#Base.rotr90(t::PEPSTensor) = permute(t, ((1,), (5, 2, 3, 4)))
Base.rot180(t::PEPSTensor) = permute(t, ((1,), (4, 5, 2, 3)))

function rot_inv(x)
    return 0.25 * (
        x +
        _make_it_fit(rotl90(x), x) +
        _make_it_fit(rot180(x), x) +
        _make_it_fit(rotr90(x), x)
    )
end

## PEPS unit cell symmetrization

PEPSLike = Union{InfinitePEPS,AbstractArray{<:PEPSTensor,2}}

symmetrize(p::PEPSLike, ::None) = p

function symmetrize(p::PEPSLike, ::Depth)
    depth, width = size(p)
    if mod(depth, 2) == 1
        for w in 1:width
            p[ceil(Int, depth / 2), w] = herm_depth_inv(p[ceil(Int, depth / 2), w])
        end
    end
    for d in 1:floor(Int, depth / 2)
        for w in 1:width
            p[depth - d + 1, w] = _make_it_fit(herm_depth(p[d, w]), p[depth - d + 1, w])
        end
    end
    return p
end

function symmetrize(p::PEPSLike, ::Width)
    depth, width = size(p)
    if mod(width, 2) == 1
        for d in 1:depth
            p[d, ceil(Int, width / 2)] = herm_width_inv(p[d, ceil(Int, width / 2), h])
        end
    end
    for w in 1:floor(Int, width / 2)
        for d in 1:depth
            p[d, width - w + 1] = _make_it_fit(herm_width(p[d, w]), p[d, width - w + 1])
        end
    end
    return p
end

function symmetrize(p::PEPSLike, ::Rot)
    return error("TODO")
end

function symmetrize(p::PEPSLike, ::Full)
    # TODO: clean up this mess...

    # some auxiliary transformations
    function symmetrize_corner(x::PEPSTensor)
        return 0.5 * (x + _make_it_fit(permute(x', ((5,), (4, 3, 2, 1))), x))
    end
    symmetrize_center(x::PEPSTensor) = herm_depth_inv(rot_inv(x))
    function symmetrize_mid_depth(x::PEPSTensor)
        return x + _make_it_fit(permute(x', ((5,), (3, 2, 1, 4))), x)
    end

    depth, width = size(p)
    depth == width || error("This only works for square unit cells.")

    odd = mod(depth, 2)
    if odd == 1
        p[ceil(Int, depth / 2), ceil(Int, width / 2)] = symmetrize_center(
            p[ceil(Int, depth / 2), ceil(Int, width / 2)]
        )
    end
    for d in 1:ceil(Int, depth / 2)
        for w in 1:floor(Int, width / 2)
            if d == w
                p[d, w] = symmetrize_corner(p[d, w])
                p[d, width - w + 1] = _make_it_fit(rotr90(p[d, w]), p[d, width - w + 1])
                p[depth - d + 1, w] = _make_it_fit(herm_depth(p[d, w]), p[depth - d + 1, w])
                p[depth - d + 1, width - w + 1] = _make_it_fit(
                    herm_depth(rotr90(p[d, w])), p[depth - d + 1, width - w + 1]
                )

            elseif odd == 1 && d == ceil(Int, depth / 2)
                p[d, w] = symmetrize_mid_depth(p[d, w])
                p[w, d] = _make_it_fit(rotr90(p[d, w]), p[w, d])
                p[d, width - w + 1] = _make_it_fit(rot180(p[d, w]), p[d, width - w + 1])
                p[width - w + 1, d] = _make_it_fit(
                    herm_depth(rotr90(p[d, w])), p[width - w + 1, d]
                )

            else
                p[depth - d + 1, w] = _make_it_fit(herm_depth(p[d, w]), p[depth - d + 1, w])
                p[w, depth - d + 1] = _make_it_fit(rotr90(p[d, w]), p[w, depth - d + 1])
                p[width - w + 1, depth - d + 1] = _make_it_fit(
                    herm_depth(rotr90(p[d, w])), [width - w + 1, depth - d + 1]
                )
                p[w, d] = _make_it_fit(rotr90(herm_depth(p[d, w])), p[w, d])
                p[width - w + 1, d] = _make_it_fit(
                    herm_depth(rotr90(herm_depth(p[d, w]))), p[width - w + 1, d]
                )
                p[d, width - w + 1] = _make_it_fit(
                    rotr90(rotr90(herm_depth(p[d, w]))), p[d, width - w + 1]
                )
                p[depth - d + 1, width - w + 1] = _make_it_fit(
                    herm_depth(rotr90(rotr90(herm_depth(p[d, w])))),
                    p[depth - d + 1, width - w + 1],
                )
            end
        end
    end
    return p
end
