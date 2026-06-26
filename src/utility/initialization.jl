"""
$(TYPEDEF)

Abstract super type for different initialization strategies for contraction environments.
"""
abstract type InitializationStyle end

"""
$(TYPEDEF)

Initialize a contraction environment from a product state made up of `(N, 0)` tensors.

## Constructors

    ProductStateInitialization(f = ones)

Contructs a product state initialization strategy, where the product state tensors
are initialized by the function `f` as `f(T::Type{<:Number}, V::ProductSpace)`.
"""
struct ProductStateInitialization{F} <: InitializationStyle
    f::F
    ProductStateInitialization(f::F = ones) where {F} = new{F}(f)
end

"""
$(TYPEDEF)

Initialize a fully random contraction environment.

## Constructors

    RandomInitialization(f = randn)

Contructs a random initialization strategy, where the environment tensors are initialized by
the function `f` as `f(T::Type{<:Number}, V::HomSpace)`.
"""
struct RandomInitialization{F} <: InitializationStyle
    f::F
    RandomInitialization(f::F = randn) where {F} = new{F}(f)
end

"""
$(TYPEDEF)

Initialize a contraction environment by applying a single iteration of a contraction
algorithm to a given environment.

## Constructors

    ApplicationInitialization(f = ones)

Contructs an application initialization strategy, where by default the starting environment
is initialized using a `ProductStateInitialization(f)` strategy.
"""
struct ApplicationInitialization{F} <: InitializationStyle
    f::F
    ApplicationInitialization(f::F = ones) where {F} = new{F}(f)
end

"""
$(TYPEDEF)

Initialize a contraction environment 

Only works in very specific cases.
"""
struct IdentityInitialization <: InitializationStyle end
