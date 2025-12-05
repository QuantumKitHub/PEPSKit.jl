"""
$(TYPEDEF)

Abstract super type for the truncation algorithms of
virtual spaces in InfinitePEPO or InfinitePEPS.
"""
abstract type ApproxAlgorithm end

function _check_virtual_dualness(state::Union{InfinitePEPS, InfinitePEPO})
    Nr, Nc = size(state)
    if isa(state, InfinitePEPO)
        @assert size(state, 3) == 1
    end
    flip_xs = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        return isdual(virtualspace(state[r, c], EAST))
    end
    flip_ys = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        return isdual(virtualspace(state[r, c], NORTH))
    end
    return flip_xs, flip_ys
end

function _standardize_virtual_spaces!(
        state::Union{InfinitePEPS, InfinitePEPO},
        flip_xs::AbstractMatrix{Bool}, flip_ys::AbstractMatrix{Bool};
        inv::Bool = false
    )
    Nr, Nc = size(flip_xs)
    for r in 1:Nr, c in 1:Nc
        inds = [
            flip_ys[r, c], flip_xs[r, c], 
            flip_ys[_next(r, Nr), c], flip_xs[r, _prev(c, Nc)]
        ]
        state.A[r, c] = flip(state.A[r, c], inds; inv)
    end
    return state
end

"""
Flip all north and east virtual spaces of `state` to non-dual spaces.
A new state is constructed.
"""
function standardize_virtual_spaces(
        state::Union{InfinitePEPS, InfinitePEPO}; inv::Bool = false
    )
    flip_xs, flip_ys = _check_virtual_dualness(state)
    !all(flip_xs) && !all(flip_ys) && (return state)
    return _standardize_virtual_spaces!(deepcopy(state), flip_xs, flip_ys; inv)
end
"""
Flip all north and east virtual spaces of `state` to non-dual spaces.
Changes are in place.
"""
function standardize_virtual_spaces!(
        state::Union{InfinitePEPS, InfinitePEPO}; inv::Bool = false
    )
    flip_xs, flip_ys = _check_virtual_dualness(state)
    !all(flip_xs) && !all(flip_ys) && (return state)
    return _standardize_virtual_spaces!(state, flip_xs, flip_ys; inv)
end
