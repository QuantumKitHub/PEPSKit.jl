"""
$(TYPEDEF)

Abstract super type for approximation algorithms
for two-dimensional tensor networks
"""
abstract type ApproximateAlgorithm end

"""
Checks if the east and the north virtual space of `state`
follow the standard convention, i.e. are dual spaces.
"""
function _check_virtual_dualness(state::Union{InfinitePEPS, InfinitePEPO})
    isdual_easts = map(CartesianIndices(state.A)) do idx
        return isdual(virtualspace(state[idx], EAST))
    end
    isdual_norths = map(CartesianIndices(state.A)) do idx
        return isdual(virtualspace(state[idx], NORTH))
    end
    return isdual_easts, isdual_norths
end
