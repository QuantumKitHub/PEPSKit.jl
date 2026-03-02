#TODO: Add docs and figure
struct CTMRGEnvTriangular{TC, TA, TB}
    "6 x rows x cols array of corner C tensors, where the first dimension specifies the spatial direction"
    C::Array{TC, 3}
    "6 x rows x cols array of edge Ta tensors, where the first dimension specifies the spatial direction"
    Ea::Array{TA, 3}
    "6 x rows x cols array of edge Tb tensors, where the first dimension specifies the spatial direction"
    Eb::Array{TB, 3}
end

function get_Ds(D::A) where {A <: ProductSpace}
    return [dir > 3 ? _elementwise_dual(D) : D for dir in 1:6]
end

function get_Ds(D::A) where {A <: ElementarySpace}
    return [dir > 3 ? ProductSpace(D') : ProductSpace(D) for dir in 1:6]
end

function CTMRGEnvTriangular(
        f, T, D::Union{A, B}, chis::B; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {A <: ProductSpace, B <: ElementarySpace}
    Ds = get_Ds(D)
    N = length(Ds[1])
    st = spacetype(Ds[1])

    T_type = tensormaptype(st, N + 1, 1, T)

    Cs = Array{T_type}(undef, 6, unitcell...)
    Eas = Array{T_type}(undef, 6, unitcell...)
    Ebs = Array{T_type}(undef, 6, unitcell...)

    for dir in 1:6, r in 1:unitcell[1], c in 1:unitcell[2]
        C = _edge_tensor(f, T, chis, Ds[dir], chis)
        Ea = _edge_tensor(f, T, chis, Ds[dir], chis)
        Eb = _edge_tensor(f, T, chis, Ds[mod1(dir + 1, 6)], chis)

        C /= norm(C)
        Ea /= norm(Ea)
        Eb /= norm(Eb)

        Cs[dir, r, c] = C
        Eas[dir, r, c] = Ea
        Ebs[dir, r, c] = Eb
    end
    return CTMRGEnvTriangular(Cs, Eas, Ebs)
end

Base.size(env::CTMRGEnvTriangular, args...) = size(env.C, args...)
Base.axes(x::CTMRGEnvTriangular, args...) = axes(x.C, args...)
function eachcoordinate(x::CTMRGEnvTriangular)
    return collect(Iterators.product(axes(x, 2), axes(x, 3)))
end
function eachcoordinate(x::CTMRGEnvTriangular, dirs)
    return collect(Iterators.product(dirs, axes(x, 2), axes(x, 3)))
end
Base.real(env::CTMRGEnvTriangular) = CTMRGEnvTriangular(real.(env.C), real.(env.Ea), real.(env.Eb))
Base.complex(env::CTMRGEnvTriangular) = CTMRGEnvTriangular(complex.(env.C), complex.(env.Ea), complex.(env.Eb))

cornertype(env::CTMRGEnvTriangular) = cornertype(typeof(env))
cornertype(::Type{CTMRGEnvTriangular{T}}) where {T} = T
edgetype(env::CTMRGEnvTriangular) = edgetype(typeof(env))
edgetype(::Type{CTMRGEnvTriangular{T}}) where {T} = T

TensorKit.spacetype(::Type{E}) where {E <: CTMRGEnvTriangular} = spacetype(cornertype(E))

# In-place update of environment
function update!(env::CTMRGEnvTriangular{T}, env´::CTMRGEnvTriangular{T}) where {T}
    env.C .= env´.C
    env.Ea .= env´.Ea
    env.Eb .= env´.Eb
    return env
end

# Custom adjoint for CTMRGEnvTriangular constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(
        ::Type{CTMRGEnvTriangular}, C::Array{T, 3}, Ea::Array{T, 3}, Eb::Array{T, 3}
    ) where {T}
    function triangularenv_pullback(Δenv_)
        Δenv = unthunk(Δenv_)
        return NoTangent(), Δenv.C, Δenv.Ea, Δenv.Eb
    end
    return CTMRGEnvTriangular(C, Ea, Eb), triangularenv_pullback
end

# math basics used for tangent accumulation in backwards pass
function Base.:+(e₁::CTMRGEnvTriangular, e₂::CTMRGEnvTriangular)
    return CTMRGEnvTriangular(e₁.C + e₂.C, e₁.Ea + e₂.Ea, e₁.Eb + e₂.Eb)
end
function Base.:-(e₁::CTMRGEnvTriangular, e₂::CTMRGEnvTriangular)
    return CTMRGEnvTriangular(e₁.C - e₂.C, e₁.Ea - e₂.Ea, e₁.Eb - e₂.Eb)
end
Base.:*(α::Number, e::CTMRGEnvTriangular) = CTMRGEnvTriangular(α * e.C, α * e.Ea, α * e.Eb)

# Custom adjoint for CTMRGEnvTriangular getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::CTMRGEnvTriangular, name::Symbol)
    result = getproperty(e, name)
    if name === :C
        function C_pullback(ΔC_)
            ΔC = unthunk(ΔC_)
            return NoTangent(), CTMRGEnvTriangular(ΔC, zerovector(e.Ea), zerovector(e.Eb))
        end
        return result, C_pullback
    elseif name === :Ea
        function Ea_pullback(ΔEa_)
            ΔEa = unthunk(ΔEa_)
            return NoTangent(), CTMRGEnvTriangular(zerovector(e.C), ΔEa, zerovector(e.Eb))
        end
        return result, Ea_pullback
    elseif name === :Eb
        function Eb_pullback(ΔEb_)
            ΔEb = unthunk(ΔEb_)
            return NoTangent(), CTMRGEnvTriangular(zerovector(e.C), zerovector(e.Ea), ΔEb)
        end
        return result, Eb_pullback
    else
        # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
end
