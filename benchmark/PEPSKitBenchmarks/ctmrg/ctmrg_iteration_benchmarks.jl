struct CTMRGSpec{S <: ElementarySpace}
    unitcell::Tuple{Int, Int}
    Pspaces::Matrix{S}
    Nspaces::Matrix{S}
    Espaces::Matrix{S}
    chi_north::Matrix{S}
    chi_east::Matrix{S}
    chi_south::Matrix{S}
    chi_west::Matrix{S}
end

_dimtag(spaces) = join(sort!(unique(dim(V) for V in spaces)), "-")

function benchname(spec::CTMRGSpec)
    rows, cols = spec.unitcell
    Dtag = _dimtag(Iterators.flatten((spec.Nspaces, spec.Espaces)))
    chitag = _dimtag(
        Iterators.flatten((spec.chi_north, spec.chi_east, spec.chi_south, spec.chi_west))
    )
    return "$(rows)x$(cols)_D$(Dtag)_chi$(chitag)"
end

function setup_problem(spec::CTMRGSpec; T::Type = ComplexF64)
    peps = InfinitePEPS(randn, T, spec.Pspaces, spec.Nspaces, spec.Espaces)
    env = CTMRGEnv(
        randn, T, peps,
        spec.chi_north, spec.chi_east, spec.chi_south, spec.chi_west,
    )
    network = InfiniteSquareNetwork(peps)
    return network, env
end

function ctmrg_iteration_benchmark(spec::CTMRGSpec, alg; T::Type = ComplexF64)
    network, env = setup_problem(spec; T)
    return @benchmarkable PEPSKit.ctmrg_iteration($network, $env, $alg)
end

# Convert a TOML value (scalar string or 2D array of strings) to a Matrix sized to `unitcell`.
function _to_space_matrix(val::AbstractString, unitcell::Tuple{Int, Int}, key::AbstractString)
    V = untomlify(VectorSpace, val)
    return fill(V, unitcell)
end
function _to_space_matrix(val::AbstractVector, unitcell::Tuple{Int, Int}, key::AbstractString)
    rows, cols = unitcell
    length(val) == rows || throw(
        ArgumentError(
            "Field `$key` has $(length(val)) rows but unitcell expects $rows."
        )
    )
    for (r, row) in enumerate(val)
        length(row) == cols || throw(
            ArgumentError(
                "Field `$key` row $r has $(length(row)) cols but unitcell expects $cols."
            )
        )
    end
    return [untomlify(VectorSpace, val[r][c]) for r in 1:rows, c in 1:cols]
end

# Resolve a per-direction matrix by trying `direct_keys` first, then falling back to `fallback_keys`.
function _resolve_matrix(d, unitcell, direct_keys, fallback_keys, label)
    for k in direct_keys
        haskey(d, k) && return _to_space_matrix(d[k], unitcell, k)
    end
    for k in fallback_keys
        haskey(d, k) && return _to_space_matrix(d[k], unitcell, k)
    end
    throw(
        ArgumentError(
            "Spec is missing `$label`: provide one of $(direct_keys) or a fallback in $(fallback_keys)."
        )
    )
end

function tomlify(spec::CTMRGSpec)
    return Dict(
        "unitcell" => collect(spec.unitcell),
        "Pspaces" => _matrix_tomlify(spec.Pspaces),
        "Nspaces" => _matrix_tomlify(spec.Nspaces),
        "Espaces" => _matrix_tomlify(spec.Espaces),
        "chi_north" => _matrix_tomlify(spec.chi_north),
        "chi_east" => _matrix_tomlify(spec.chi_east),
        "chi_south" => _matrix_tomlify(spec.chi_south),
        "chi_west" => _matrix_tomlify(spec.chi_west),
    )
end

_matrix_tomlify(M::AbstractMatrix{<:ElementarySpace}) =
    [[tomlify(M[r, c]) for c in axes(M, 2)] for r in axes(M, 1)]

function untomlify(::Type{CTMRGSpec}, d)
    unitcell = (Int(d["unitcell"][1]), Int(d["unitcell"][2]))

    Pspaces = _resolve_matrix(d, unitcell, ("Pspaces", "Pspace"), (), "physical space")
    Nspaces = _resolve_matrix(d, unitcell, ("Nspaces", "Dspace"), (), "north virtual space")
    Espaces = _resolve_matrix(d, unitcell, ("Espaces", "Dspace"), ("Nspaces",), "east virtual space")

    chi_north = _resolve_matrix(d, unitcell, ("chi_north", "chispace"), (), "north environment space")
    chi_east = _resolve_matrix(d, unitcell, ("chi_east", "chispace"), ("chi_north",), "east environment space")
    chi_south = _resolve_matrix(d, unitcell, ("chi_south", "chispace"), ("chi_north",), "south environment space")
    chi_west = _resolve_matrix(d, unitcell, ("chi_west", "chispace"), ("chi_north",), "west environment space")

    return CTMRGSpec(
        unitcell,
        Pspaces, Nspaces, Espaces,
        chi_north, chi_east, chi_south, chi_west,
    )
end
