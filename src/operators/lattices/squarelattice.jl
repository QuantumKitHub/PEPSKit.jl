"""
    InfiniteSquare(Nrows::Integer=1, Ncols::Integer=1)

Infinite square lattice with a unit cell of size `(Nrows, Ncols)`.
"""
struct InfiniteSquare <: AbstractLattice{2}
    Nrows::Int
    Ncols::Int
    function InfiniteSquare(Nrows::Integer=1, Ncols::Integer=1)
        Nrows > 0 && Ncols > 0 || error("unit cell size needs to be positive")
        return new(Nrows, Ncols)
    end
end

function vertices(lattice::InfiniteSquare)
    return CartesianIndices((1:(lattice.Nrows), 1:(lattice.Ncols)))
end

"""
    nearest_neighbours(lattice::InfiniteSquare)

Return the nearest neighbors of the lattice `lattice`.

````
    +---*---+
    |   
    *
    |
    +
````
"""
function nearest_neighbours(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex}[]
    for idx in vertices(lattice)
        push!(neighbors, (idx, idx + CartesianIndex(0, 1)))
        push!(neighbors, (idx, idx + CartesianIndex(1, 0)))
    end
    return neighbors
end

"""
    next_nearest_neighbours(lattice::InfiniteSquare)

Return the next nearest neighbors of the lattice `lattice`.

````
    +------+
    | \\ ╱ | 
    |   *  |
    | ╱  \\|
    +------+
````
"""
function next_nearest_neighbours(lattice::InfiniteSquare)
    neighbors = Tuple{CartesianIndex,CartesianIndex}[]
    for idx in vertices(lattice)
        push!(neighbors, (idx, idx + CartesianIndex(1, 1)))
        push!(neighbors, (idx + CartesianIndex(0, 1), idx + CartesianIndex(1, 0)))
    end
    return neighbors
end
