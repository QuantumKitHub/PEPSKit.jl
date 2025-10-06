abstract type InitializationStyle end
struct ProductStateInitialization <: InitializationStyle end
struct RandomInitialization{F} <: InitializationStyle
    f::F
    RandomInitialization(f::F = randn) where {F} = new{F}(f)
end
struct ApplicationInitialization <: InitializationStyle end

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::RandomInitialization,
        virtual_spaces... = oneunit(spacetype(n)),
    )
    return CTMRGEnv(alg.f, elt, n, virtual_spaces...)
end

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::ProductStateInitialization,
        virtual_spaces... = oneunit(spacetype(n)),
    )
    i = one(sectortype(n))
    env = CTMRGEnv(ones, elt, n, virtual_spaces...)
    for (dir, r, c) in Iterators.product(axes(env)...)
        @assert i in blocksectors(env.corners[dir, r, c])
        for (c, b) in blocks(env.corners[dir, r, c])
            b .= 0
            c == i && (b[1, 1] = 1)
        end
    end
    return env
end

function initialize_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::ApplicationInitialization,
        trscheme::TruncationScheme;
        boundary_alg = (;
            alg = :sequential, tol = 1.0e-5, maxiter = 10, verbosity = -1,
        )
    )
    boundary_alg = (; boundary_alg..., trscheme) # merge trscheme with optional alg definition
    env = initialize_environment(elt, n, ProductStateInitialization())
    env, = leading_boundary(env, n; boundary_alg...)
    return env
end

function initialize_environment(n::InfiniteSquareNetwork, args...; kwargs...)
    return initialize_environment(scalartype(n), n, args...; kwargs...)
end
function initialize_environment(A::Union{InfinitePEPS, InfinitePartitionFunction}, args...; kwargs...)
    return initialize_environment(scalartype(A), A, args...; kwargs...)
end
function initialize_environment(elt::Type{<:Number}, A::Union{InfinitePEPS, InfinitePartitionFunction}, args...; kwargs...)
    return initialize_environment(elt, InfiniteSquareNetwork(A), args...; kwargs...)
end
