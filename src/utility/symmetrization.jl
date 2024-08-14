abstract type SymmetrizationStyle end

"""
    struct ReflectDepth <: SymmetrizationStyle

Reflection symmmetrization along the horizontal axis, such that north and south are mirrored.
"""
struct ReflectDepth <: SymmetrizationStyle end

"""
    struct ReflectWidth <: SymmetrizationStyle

Reflection symmmetrization along the vertical axis, such that east and west are mirrored.
"""
struct ReflectWidth <: SymmetrizationStyle end

# TODO
struct Rotate <: SymmetrizationStyle end

"""
    struct RotateReflect <: SymmetrizationStyle

Full reflection and rotation symmmetrization, such that reflection along the horizontal and
vertical axis as well as π/2 rotations leave the object invariant.
"""
struct RotateReflect <: SymmetrizationStyle end

# some rather shady definitions for 'hermitian conjugate' at the level of a single tensor

function herm_depth(x::PEPSTensor)
    return permute(x', ((5,), (3, 2, 1, 4)))
end
function herm_depth(x::PEPOTensor)
    return permute(x', ((5, 6), (3, 2, 1, 4)))
end

function herm_width(x::PEPSTensor)
    return permute(x', ((5,), (1, 4, 3, 2)))
end
function herm_width(x::PEPOTensor)
    return permute(x', ((5, 6), (1, 4, 3, 2)))
end

function herm_height(x::PEPOTensor)
    return permute(x', ((6, 5), (1, 2, 3, 4)))
end

# hermitian invariance

# make two TensorMap's have the same spaces, by force if necessary
# this is definitely not what you would want to do, but it circumvents having to think
# about what hermiticity means at the level of transfer operators, which is something
function _fit_spaces(
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
    return 0.5 * (x + _fit_spaces(herm_depth(x), x))
end

function herm_width_inv(x::Union{PEPSTensor,PEPOTensor})
    return 0.5 * (x + _fit_spaces(herm_width(x), x))
end

function herm_height_inv(x::PEPOTensor)
    return 0.5 * (x + _fit_spaces(herm_height(x), x))
end

# rotation invariance

function rot_inv(x)
    return 0.25 * (
        x +
        _fit_spaces(rotl90(x), x) +
        _fit_spaces(rot180(x), x) +
        _fit_spaces(rotr90(x), x)
    )
end

# PEPS unit cell symmetrization

"""
    symmetrize!(peps::InfinitePEPS, ::SymmetrizationStyle)

Symmetrize a PEPS using the given `SymmetrizationStyle` in-place.
"""
symmetrize!(peps::InfinitePEPS, ::Nothing) = peps

function symmetrize!(peps::InfinitePEPS, ::ReflectDepth)
    depth, width = size(peps)
    if mod(depth, 2) == 1
        for w in 1:width
            peps[ceil(Int, depth / 2), w] = herm_depth_inv(peps[ceil(Int, depth / 2), w])
        end
    end
    for d in 1:floor(Int, depth / 2)
        for w in 1:width
            peps[depth - d + 1, w] = _fit_spaces(
                herm_depth(peps[d, w]), peps[depth - d + 1, w]
            )
        end
    end
    return peps
end

function symmetrize!(peps::InfinitePEPS, ::ReflectWidth)
    depth, width = size(peps)
    if mod(width, 2) == 1
        for d in 1:depth
            peps[d, ceil(Int, width / 2)] = herm_width_inv(peps[d, ceil(Int, width / 2)])
        end
    end
    for w in 1:floor(Int, width / 2)
        for d in 1:depth
            peps[d, width - w + 1] = _fit_spaces(
                herm_width(peps[d, w]), peps[d, width - w + 1]
            )
        end
    end
    return peps
end

function symmetrize!(peps::InfinitePEPS, ::Rotate)
    return error("TODO")
end

function symmetrize!(peps::InfinitePEPS, symm::RotateReflect)
    # TODO: clean up this mess...

    # some auxiliary transformations
    function symmetrize_corner(x::PEPSTensor)
        return 0.5 * (x + _fit_spaces(permute(x', ((5,), (4, 3, 2, 1))), x))
    end
    symmetrize_center(x::PEPSTensor) = herm_depth_inv(rot_inv(x))
    function symmetrize_mid_depth(x::PEPSTensor)
        return x + _fit_spaces(permute(x', ((5,), (3, 2, 1, 4))), x)
    end

    depth, width = size(peps)
    depth == width || ArgumentError("$symm can only be applied to square unit cells")

    odd = mod(depth, 2)
    if odd == 1
        peps[ceil(Int, depth / 2), ceil(Int, width / 2)] = symmetrize_center(
            peps[ceil(Int, depth / 2), ceil(Int, width / 2)]
        )
    end
    for d in 1:ceil(Int, depth / 2)
        for w in 1:floor(Int, width / 2)
            if d == w
                peps[d, w] = symmetrize_corner(peps[d, w])
                peps[d, width - w + 1] = _fit_spaces(
                    rotr90(peps[d, w]), peps[d, width - w + 1]
                )
                peps[depth - d + 1, w] = _fit_spaces(
                    herm_depth(peps[d, w]), peps[depth - d + 1, w]
                )
                peps[depth - d + 1, width - w + 1] = _fit_spaces(
                    herm_depth(rotr90(peps[d, w])), peps[depth - d + 1, width - w + 1]
                )

            elseif odd == 1 && d == ceil(Int, depth / 2)
                peps[d, w] = symmetrize_mid_depth(peps[d, w])
                peps[w, d] = _fit_spaces(rotr90(peps[d, w]), peps[w, d])
                peps[d, width - w + 1] = _fit_spaces(
                    rot180(peps[d, w]), peps[d, width - w + 1]
                )
                peps[width - w + 1, d] = _fit_spaces(
                    herm_depth(rotr90(peps[d, w])), peps[width - w + 1, d]
                )

            else
                peps[depth - d + 1, w] = _fit_spaces(
                    herm_depth(peps[d, w]), peps[depth - d + 1, w]
                )
                peps[w, depth - d + 1] = _fit_spaces(
                    rotr90(peps[d, w]), peps[w, depth - d + 1]
                )
                peps[width - w + 1, depth - d + 1] = _fit_spaces(
                    herm_depth(rotr90(peps[d, w])), [width - w + 1, depth - d + 1]
                )
                peps[w, d] = _fit_spaces(rotr90(herm_depth(peps[d, w])), peps[w, d])
                peps[width - w + 1, d] = _fit_spaces(
                    herm_depth(rotr90(herm_depth(peps[d, w]))), peps[width - w + 1, d]
                )
                peps[d, width - w + 1] = _fit_spaces(
                    rotr90(rotr90(herm_depth(peps[d, w]))), peps[d, width - w + 1]
                )
                peps[depth - d + 1, width - w + 1] = _fit_spaces(
                    herm_depth(rotr90(rotr90(herm_depth(peps[d, w])))),
                    peps[depth - d + 1, width - w + 1],
                )
            end
        end
    end
    return peps
end

"""
    symmetrize_callback(peps, envs, E, grad, symm::SymmetrizationStyle)

Callback function symmetrizing both the `peps` and `grad` tensors.
"""
function symmetrize_callback(peps, envs, E, grad, symm::SymmetrizationStyle)
    peps_symm = symmetrize!(deepcopy(peps), symm)
    grad_symm = symmetrize!(deepcopy(grad), symm)
    return peps_symm, envs, E, grad_symm
end
