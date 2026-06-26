"""
    initialize_ctmrg_environment([elt::Type{<:Number},] n::InfiniteSquareNetwork, alg::RandomInitialization, virtual_spaces...)

Initialize a fully random `CTMRGEnv` using the given environment virtual spaces. See
[`CTMRGEnv`](@ref) for details on the expected format of the virtual spaces.
"""
function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::RandomInitialization,
        virtual_spaces... = oneunit(spacetype(n)),
    )
    return CTMRGEnv(alg.f, elt, n, virtual_spaces...)
end

"""
    initialize_ctmrg_environment([elt::Type{<:Number},] n::InfiniteSquareNetwork, alg::RandomInitialization)

Initialize a `CTMRGEnv` corresponding to a product state with trivial virtual spaces and
corners. The product state edge tensors are initialized as `alg.f(elt, V::ProductSpace)`.
"""
function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::ProductStateInitialization,
    )
    env = CTMRGEnv(ProductStateEnv(alg.f, elt, n))
    return env
end

"""
    initialize_ctmrg_environment([elt::Type{<:Number},] n::InfiniteSquareNetwork, alg::RandomInitialization, [env0])

Initialize a `CTMRGEnv` by applying a single untruncated iteration of
[`SimultaneousCTMRG`](@ref) to a given initial environment. By default, the starting
environment is chosen as a random product state.
"""
function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        alg::ApplicationInitialization,
        env0 = ProductStateEnv(alg.f, elt, n)
    )
    dummy_alg = SimultaneousCTMRG(trunc = (; alg = :notrunc))
    env, = ctmrg_iteration(n, CTMRGEnv(env0), dummy_alg)
    return env
end

_check_two_layer(::InfiniteSquareNetwork) = false
_check_two_layer(::InfiniteSquareNetwork{<:PEPSSandwich}) = true

"""
    initialize_ctmrg_environment([elt::Type{<:Number},] n::InfiniteSquareNetwork, alg::RandomInitialization, [env0])

Initialize a `CTMRGEnv` corresponding to a product state acting as an identity between the
virtual spaces of a two-layer network, for example
```
         ╱      
┌-----ket----- 
|    ╱ |        
|      |   
|      | ╱      
└-----bra----- 
     ╱          
```
"""
function initialize_ctmrg_environment(
        elt::Type{<:Number},
        n::InfiniteSquareNetwork,
        ::IdentityInitialization,
    )
    _check_two_layer(n) ||
        throw(ArgumentError("Identity initialization is only defined for two-layer networks."))
    bp_env = BPEnv(isomorphism, elt, n)
    env = CTMRGEnv(bp_env)
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
