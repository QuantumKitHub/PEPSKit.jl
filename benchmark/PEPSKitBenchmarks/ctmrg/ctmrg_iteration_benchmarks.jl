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

function benchname(spec::CTMRGSpec)
    rows, cols = spec.unitcell
    Dmin = min(minimum(dim, spec.Nspaces), minimum(dim, spec.Espaces))
    chimin = min(
        minimum(dim, spec.chi_north), minimum(dim, spec.chi_east),
        minimum(dim, spec.chi_south), minimum(dim, spec.chi_west)
    )
    return "$(rows)x$(cols)_D$(Dmin)_chi$(chimin)"
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

function _get_matrix(d, key, unitcell)
    haskey(d, key) || throw(ArgumentError("Spec is missing required key `$key`."))
    return _to_space_matrix(d[key], unitcell, key)
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

    Pspaces = _get_matrix(d, "Pspaces", unitcell)
    Nspaces = _get_matrix(d, "Nspaces", unitcell)
    Espaces = _get_matrix(d, "Espaces", unitcell)

    chi_north = _get_matrix(d, "chi_north", unitcell)
    chi_east = _get_matrix(d, "chi_east", unitcell)
    chi_south = _get_matrix(d, "chi_south", unitcell)
    chi_west = _get_matrix(d, "chi_west", unitcell)

    return CTMRGSpec(
        unitcell,
        Pspaces, Nspaces, Espaces,
        chi_north, chi_east, chi_south, chi_west,
    )
end
