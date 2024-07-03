# Contraction of local operators on arbitrary lattice locations
# -------------------------------------------------------------
import MPSKit: tensorexpr

# currently need this because MPSKit restricts tensor names to symbols
MPSKit.tensorexpr(ex::Expr, inds) = Expr(:ref, ex, inds...)
function MPSKit.tensorexpr(ex::Expr, indout, indin)
    return Expr(:typed_vcat, ex, Expr(:row, indout...), Expr(:row, indin...))
end

"""
    contract_localoperator(inds, O, peps, env)

Contract a local operator `O` on the PEPS `peps` at the indices `inds` using the environment `env`.
"""
function contract_localoperator(
    inds::NTuple{N,CartesianIndex{2}},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    static_inds = Val.(inds)
    return _contract_localoperator(static_inds, O, ket, bra, env)
end
function contract_localoperator(
    inds::NTuple{N,Tuple{Int,Int}},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    return contract_localoperator(CartesianIndex.(inds), O, ket, bra, env)
end

# settings for determining contraction orders
const PEPS_PHYSICALDIM = 2
const PEPS_BONDDIM = :χ
const PEPS_ENVBONDDIM = :(χ^2)

# This implements the contraction of an operator acting on sites `inds`. 
# The generated function ensures that we can use @tensor to write dynamic contractions (and maximize performance).
@generated function _contract_localoperator(
    inds::NTuple{N,Val},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    if !allunique(cartesian_inds)
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    end

    xmin, xmax = extrema(getindex.(cartesian_inds, 1))
    ymin, ymax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (xmax - xmin + 1, ymax - ymin + 1)

    corner_NW = tensorexpr(
        :(env.corners[NORTHWEST, mod1($(xmin), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_NW_1,),
        (:C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[NORTHEAST, mod1($(xmin), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_NE_1,),
        (:C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[SOUTHEAST, mod1($(xmax), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_SE_1,),
        (:C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[SOUTHWEST, mod1($(xmax), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_SW_1,),
        (:C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH, mod1($(xmin), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NW_2 : Symbol(:E_N_virtual, i - 1)),
                Symbol(:E_N_top, i),
                Symbol(:E_N_bot, i),
            ),
            ((i == gridsize[2] ? :C_NE_1 : Symbol(:E_N_virtual, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymax), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NE_2 : Symbol(:E_E_virtual, i - 1)),
                Symbol(:E_E_top, i),
                Symbol(:E_E_bot, i),
            ),
            ((i == gridsize[1] ? :C_SE_1 : Symbol(:E_E_virtual, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH, mod1($(xmax), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == gridsize[2] ? :C_SE_2 : Symbol(:E_S_virtual, i)),
                Symbol(:E_S_top, i),
                Symbol(:E_S_bot, i),
            ),
            ((i == 1 ? :C_SW_1 : Symbol(:E_S_virtual, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin), size(ket, 2))
            ]),
            (
                (i == gridsize[1] ? :C_SW_2 : Symbol(:E_W_virtual, i)),
                Symbol(:E_W_top, i),
                Symbol(:E_W_bot, i),
            ),
            ((i == 1 ? :C_NW_1 : Symbol(:E_W_virtual, i - 1)),),
        )
    end

    operator = tensorexpr(
        :O, ntuple(i -> Symbol(:O_out_, i), N), ntuple(i -> Symbol(:O_in_, i), N)
    )

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(xmin + i - 1, ymin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:physical, i, "_", j) : Symbol(:O_out_, inds_id)
        return tensorexpr(
            :(bra[
                mod1($(xmin + i - 1), size(bra, 1)), mod1($(ymin + j - 1), size(bra, 2))
            ]),
            (physical_label,),
            (
                (i == 1 ? Symbol(:E_N_bot, j) : Symbol(:bra_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_bot, i)
                    else
                        Symbol(:bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_bot, j)
                    else
                        Symbol(:bra_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_bot, i) : Symbol(:bra_horizontal, i, "_", j - 1)),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(xmin + i - 1, ymin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:physical, i, "_", j) : Symbol(:O_in_, inds_id)
        return tensorexpr(
            :(ket[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (physical_label,),
            (
                (i == 1 ? Symbol(:E_N_top, j) : Symbol(:ket_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_top, i)
                    else
                        Symbol(:ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_top, j)
                    else
                        Symbol(:ket_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_top, i) : Symbol(:ket_horizontal, i, "_", j - 1)),
            ),
        )
    end

    multiplication_ex = Expr(
        :call,
        :*,
        corner_NW,
        corner_NE,
        corner_SE,
        corner_SW,
        edges_N...,
        edges_E...,
        edges_S...,
        edges_W...,
        ket...,
        map(x -> Expr(:call, :conj, x), bra)...,
        operator,
    )

    opt_ex = Expr(:tuple)
    allinds = TensorOperations.getallindices(multiplication_ex)
    for label in allinds
        if startswith(String(label), "physical") || startswith(String(label), "O")
            push!(opt_ex.args, :($label => $PEPS_PHYSICALDIM))
        elseif startswith(String(label), "ket") || startswith(String(label), "bra")
            push!(opt_ex.args, :($label => $PEPS_BONDDIM))
        else
            push!(opt_ex.args, :($label => $PEPS_ENVBONDDIM))
        end
    end

    return quote
        @tensor opt = $opt_ex $multiplication_ex
    end
end

"""
    contract_localnorm(inds, peps, env)

Contract a local norm of the PEPS `peps` around indices `inds`.
"""
function contract_localnorm(
    inds::NTuple{N,CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    static_inds = Val.(inds)
    return _contract_localnorm(static_inds, ket, bra, env)
end
function contract_localnorm(
    inds::NTuple{N,Tuple{Int,Int}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    return contract_localnorm(CartesianIndex.(inds), ket, bra, env)
end
@generated function _contract_localnorm(
    inds::NTuple{N,Val}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    if !allunique(cartesian_inds)
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    end

    xmin, xmax = extrema(getindex.(cartesian_inds, 1))
    ymin, ymax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (xmax - xmin + 1, ymax - ymin + 1)

    corner_NW = tensorexpr(
        :(env.corners[NORTHWEST, mod1($(xmin), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_NW_1,),
        (:C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[NORTHEAST, mod1($(xmin), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_NE_1,),
        (:C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[SOUTHEAST, mod1($(xmax), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_SE_1,),
        (:C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[SOUTHWEST, mod1($(xmax), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_SW_1,),
        (:C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH, mod1($(xmin), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NW_2 : Symbol(:E_N_virtual, i - 1)),
                Symbol(:E_N_top, i),
                Symbol(:E_N_bot, i),
            ),
            ((i == gridsize[2] ? :C_NE_1 : Symbol(:E_N_virtual, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymax), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NE_2 : Symbol(:E_E_virtual, i - 1)),
                Symbol(:E_E_top, i),
                Symbol(:E_E_bot, i),
            ),
            ((i == gridsize[1] ? :C_SE_1 : Symbol(:E_E_virtual, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH, mod1($(xmax), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == gridsize[2] ? :C_SE_2 : Symbol(:E_S_virtual, i)),
                Symbol(:E_S_top, i),
                Symbol(:E_S_bot, i),
            ),
            ((i == 1 ? :C_SW_1 : Symbol(:E_S_virtual, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin), size(ket, 2))
            ]),
            (
                (i == gridsize[1] ? :C_SW_2 : Symbol(:E_W_virtual, i)),
                Symbol(:E_W_top, i),
                Symbol(:E_W_bot, i),
            ),
            ((i == 1 ? :C_NW_1 : Symbol(:E_W_virtual, i - 1)),),
        )
    end

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(bra[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (Symbol(:physical, i, "_", j),),
            (
                (i == 1 ? Symbol(:E_N_bot, j) : Symbol(:bra_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_bot, i)
                    else
                        Symbol(:bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_bot, j)
                    else
                        Symbol(:bra_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_bot, i) : Symbol(:bra_horizontal, i, "_", j - 1)),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(ket[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (Symbol(:physical, i, "_", j),),
            (
                (i == 1 ? Symbol(:E_N_top, j) : Symbol(:ket_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_top, i)
                    else
                        Symbol(:ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_top, j)
                    else
                        Symbol(:ket_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_top, i) : Symbol(:ket_horizontal, i, "_", j - 1)),
            ),
        )
    end

    multiplication_ex = Expr(
        :call,
        :*,
        corner_NW,
        corner_NE,
        corner_SE,
        corner_SW,
        edges_N...,
        edges_E...,
        edges_S...,
        edges_W...,
        ket...,
        map(x -> Expr(:call, :conj, x), bra)...,
    )

    opt_ex = Expr(:tuple)
    allinds = TensorOperations.getallindices(multiplication_ex)
    for label in allinds
        if startswith(String(label), "physical")
            push!(opt_ex.args, :($label => $PEPS_PHYSICALDIM))
        elseif startswith(String(label), "ket") || startswith(String(label), "bra")
            push!(opt_ex.args, :($label => $PEPS_BONDDIM))
        else
            push!(opt_ex.args, :($label => $PEPS_ENVBONDDIM))
        end
    end

    return quote
        @tensor opt = $opt_ex $multiplication_ex
    end
end

# Hamiltonian consisting of local terms
# -------------------------------------
struct PEPSHamiltonian{T<:Tuple,S}
    lattice::Matrix{S}
    terms::T
end
function PEPSHamiltonian(lattice::Matrix{S}, terms::Pair...) where {S}
    lattice′ = PeriodicArray(lattice)
    for (inds, operator) in terms
        @assert operator isa AbstractTensorMap
        @assert numout(operator) == numin(operator) == length(inds)
        for i in 1:length(inds)
            @assert space(operator, i) == lattice′[inds[i]]
        end
    end
    return PEPSHamiltonian{typeof(terms),S}(lattice, terms)
end

"""
    checklattice(Bool, args...)
    checklattice(args...)

Helper function for checking lattice compatibility. The first version returns a boolean,
while the second version throws an error if the lattices do not match.
"""
function checklattice(args...)
    return checklattice(Bool, args...) || throw(ArgumentError("Lattice mismatch."))
end
function checklattice(::Type{Bool}, peps::InfinitePEPS, H::PEPSHamiltonian)
    return size(peps) == size(H.lattice)
end
function checklattice(::Type{Bool}, H::PEPSHamiltonian, peps::InfinitePEPS)
    return checklattice(Bool, peps, H)
end
@non_differentiable checklattice(args...)

function nearest_neighbour_hamiltonian(
    lattice::Matrix{S}, h::AbstractTensorMap{S,2,2}
) where {S}
    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        J1 = I + CartesianIndex(1, 0)
        J2 = I + CartesianIndex(0, 1)
        push!(terms, (I, J1) => h)
        push!(terms, (I, J2) => h)
    end
    return PEPSHamiltonian(lattice, terms...)
end

function Base.repeat(H::PEPSHamiltonian, m::Int, n::Int)
    lattice = repeat(H.lattice, m, n)
    terms = []
    for (inds, operator) in H.terms, i in 1:m, j in 1:n
        offset = CartesianIndex((i - 1) * size(H.lattice, 1), (j - 1) * size(H.lattice, 2))
        push!(terms, (inds .+ Ref(offset)) => operator)
    end
    return PEPSHamiltonian(lattice, terms...)
end

function MPSKit.expectation_value(peps::InfinitePEPS, H::PEPSHamiltonian, envs::CTMRGEnv)
    checklattice(peps, H)
    return sum(H.terms) do (inds, operator)
        contract_localoperator(inds, operator, peps, peps, envs) /
        contract_localnorm(inds, peps, peps, envs)
    end
end

function costfun(peps::InfinitePEPS, envs::CTMRGEnv, H::PEPSHamiltonian)
    E = MPSKit.expectation_value(peps, H, envs)
    ignore_derivatives() do
        isapprox(imag(E), 0; atol=sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end
