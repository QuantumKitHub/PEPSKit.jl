# """
# $(TYPEDEF)

# Wrapper for a QR decomposition algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
# If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.

# ## Fields

# $(TYPEDFIELDS)

# ## Constructors

#     QRAdjoint(; kwargs...)

# Construct a `QRAdjoint` algorithm struct based on the following keyword arguments:

# * `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.qr_fwd_alg))`: Eig algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where `alg` is one of the following:
# * `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.qr_rrule_alg))`: Reverse-rule algorithm for differentiating the eigenvalue decomposition. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
# """
# struct QRAdjoint{F, R}
#     fwd_alg::F
#     rrule_alg::R
# end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

# const QR_FWD_SYMBOLS = IdDict{Symbol, Any}(
#     # TODO
# )
# const QR_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
#     # TODO
# )

# function QRAdjoint(; fwd_alg = (;), rrule_alg = (;))
#     # parse forward algorithm
#     fwd_algorithm = if fwd_alg isa NamedTuple
#         fwd_kwargs = (; alg = Defaults.qr_fwd_alg, fwd_alg...) # overwrite with specified kwargs
#         haskey(QR_FWD_SYMBOLS, fwd_kwargs.alg) ||
#             throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
#         fwd_type = QR_FWD_SYMBOLS[fwd_kwargs.alg]
#         fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg = nothing)) # remove `alg` keyword argument
#         fwd_type(; fwd_kwargs...)
#     else
#         fwd_alg
#     end

#     # parse reverse-rule algorithm
#     rrule_algorithm = if rrule_alg isa NamedTuple
#         rrule_kwargs = (;
#             alg = Defaults.qr_rrule_alg,
#             verbosity = Defaults.qr_rrule_verbosity,
#             rrule_alg...,
#         ) # overwrite with specified kwargs

#         haskey(QR_RRULE_SYMBOLS, rrule_kwargs.alg) ||
#             throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
#         rrule_type = QR_RRULE_SYMBOLS[rrule_kwargs.alg]
#         if rrule_type <: Something # TODO
#             rrule_kwargs = (; rrule_kwargs.verbosity)
#         end

#         rrule_type(; rrule_kwargs...)
#     else
#         rrule_alg
#     end

#     return QRAdjoint(fwd_algorithm, rrule_algorithm)
# end

# TODO: implement wrapper for MatrixAlgebraKit QR functions
