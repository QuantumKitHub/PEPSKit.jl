abstract type InitializationStyle end
struct ProductStateInitialization{F} <: InitializationStyle
    f::F
    ProductStateInitialization(f::F = ones) where {F} = new{F}(f)
end
struct RandomInitialization{F} <: InitializationStyle
    f::F
    RandomInitialization(f::F = randn) where {F} = new{F}(f)
end
struct ApplicationInitialization{F} <: InitializationStyle
    f::F
    ApplicationInitialization(f::F = ones) where {F} = new{F}(f)
end

# initialize randomly, using same virtual space specification as the CTMRGEnv constructor
function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::RandomInitialization,
        virtual_spaces... = oneunit(spacetype(n)),
    )
    return CTMRGEnv(alg.f, elt, n, virtual_spaces...)
end

function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::ProductStateInitialization,
    )
    env = CTMRGEnv(ProductStateEnv(alg.f, elt, n))
    return env
end

function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::ApplicationInitialization,
        env0::ProductStateEnv = ProductStateEnv(alg.f, elt, n)
    )
    dummy_alg = CTMRGAlgorithm(; alg = :simultaneous, trunc = (; alg = :notrunc))
    env, = ctmrg_iteration(n, CTMRGEnv(env0), dummy_alg)
    return env
end

function initialize_ctmrg_environment(
        A::Union{InfiniteSquareNetwork, InfinitePEPS, InfinitePartitionFunction}, args...;
        kwargs...
    )
    return initialize_ctmrg_environment(scalartype(A), A, args...; kwargs...)
end
function initialize_ctmrg_environment(
        elt::Type{<:Number}, A::Union{InfinitePEPS, InfinitePartitionFunction}, args...;
        kwargs...
    )
    return initialize_ctmrg_environment(elt, InfiniteSquareNetwork(A), args...; kwargs...)
end
