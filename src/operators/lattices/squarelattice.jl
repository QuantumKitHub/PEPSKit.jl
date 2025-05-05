"""
$(TYPEDEF)

Infinite square lattice with a unit cell of size `(Nrows, Ncols)`.

## Fields

$(TYPEDFIELDS)

## Constructor

    InfiniteSquare([Nrows=1, Ncols=1])

By default, an infinite square with a (1, 1)-unitcell is constructed.
"""
struct InfiniteSquare <: AbstractLattice{2}
    Nrows::Int
    Ncols::Int
    function InfiniteSquare(Nrows::Integer=1, Ncols::Integer=1)
        Nrows > 0 && Ncols > 0 || error("unit cell size needs to be positive")
        return new(Nrows, Ncols)
    end
end

Base.size(lattice::InfiniteSquare) = (lattice.Nrows, lattice.Ncols)

function vertices(lattice::InfiniteSquare)
    return CartesianIndices((1:(lattice.Nrows), 1:(lattice.Ncols)))
end

function nearest_neighbours(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex}[]
    for idx in vertices(lattice)
        push!(neighbors, (idx, idx + CartesianIndex(0, 1)))
        push!(neighbors, (idx, idx + CartesianIndex(1, 0)))
    end
    return neighbors
end

function next_nearest_neighbours(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex}[]
    for idx in vertices(lattice)
        push!(neighbors, (idx, idx + CartesianIndex(1, 1)))
        push!(neighbors, (idx + CartesianIndex(0, 1), idx + CartesianIndex(1, 0)))
    end
    return neighbors
end
