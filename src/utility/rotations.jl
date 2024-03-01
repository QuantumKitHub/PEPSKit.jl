const NORTH = 1
const EAST = 2
const SOUTH = 3
const WEST = 4

const NORTHWEST = 1
const NORTHEAST = 2
const SOUTHEAST = 3
const SOUTHWEST = 4

# Rotate tensor to any direction by successive application of Base.rotl90
rotate_north(t, dir) = mod1(dir, 4) == NORTH ? t : rotate_north(rotl90(t), dir - 1)

# Hacked version for AbstractArray{T,3} which doesn't need to overload rotl90 to avoid type piracy
function rotate_north(A::AbstractArray{T,3}, dir) where {T}
    for _ in 1:(mod1(dir, 4) - 1)
        # Initialize copy with rotated sizes
        A′ = Array{T,3}(undef, size(A, 1), size(A, 3), size(A, 2))
        for dir in 1:size(A, 1)
            @diffset A′[_prev(dir, 4), :, :] .= rotl90(A[dir, :, :])
        end
        A = A′
    end

    return A
end
