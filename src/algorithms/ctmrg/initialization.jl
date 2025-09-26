abstract type InitializationStyle end
struct ProductStateInitialization <: InitializationStyle end
struct RandomInitialization <: InitializationStyle end
struct ApplicationInitialization <: InitializationStyle end

_to_tuple(x) = (x,)
_to_tuple(x::Tuple) = x

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::RandomInitialization,
        init_spec = oneunit(spacetype(n)),
    )
    return CTMRGEnv(randn, elt, n, _to_tuple(init_spec)...)
end

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::ProductStateInitialization,
        init_spec = oneunit(spacetype(n)),
    )
    i = one(sectortype(init_spec))
    env = CTMRGEnv(ones, elt, n, _to_tuple(init_spec)...)
    for (dir, r, c) in Iterators.product(axes(env)...)
        @assert i in blocksectors(env.corners[dir, r, c])
        block(env.corners[dir, r, c], i)[1, 1] = 1
    end
    return env
end

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::ApplicationInitialization,
        init_spec::TruncationScheme;
        boundary_alg = (;
            alg = :sequential, tol = 1.0e-5, maxiter = 10, verbosity = -1, trscheme = init_spec,
        )
    )
    env = initialize_environment(elt, n, ProductStateInitialization())
    env, = leading_boundary(env, n; boundary_alg...)
    return env
end

function initialize_environment(n::InfiniteSquareNetwork, args...; kwargs...)
    return initialize_environment(ComplexF64, n, args...; kwargs...)
end
function initialize_environment(A::Union{InfinitePEPS, InfinitePartitionFunction}, args...; kwargs...)
    return initialize_environment(ComplexF64, A, args...; kwargs...)
end
function initialize_environment(elt::Type{<:Number}, A::Union{InfinitePEPS, InfinitePartitionFunction}, args...; kwargs...)
    return initialize_environment(elt, InfiniteSquareNetwork(A), args...; kwargs...)
end
