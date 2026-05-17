struct CTMRGSpec{S <: ElementarySpace}
    unitcell::Tuple{Int, Int}
    Pspace::S
    Dspace::S
    chispace::S
end

function benchname(spec::CTMRGSpec)
    return "$(spec.unitcell[1])x$(spec.unitcell[2])_D$(dim(spec.Dspace))_chi$(dim(spec.chispace))"
end

function setup_problem(spec::CTMRGSpec; T::Type = ComplexF64)
    rows, cols = spec.unitcell
    Pspaces = fill(spec.Pspace, rows, cols)
    Nspaces = fill(spec.Dspace, rows, cols)
    Espaces = fill(spec.Dspace, rows, cols)
    chis = fill(spec.chispace, rows, cols)

    peps = InfinitePEPS(randn, T, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(randn, T, peps, chis, chis, chis, chis)
    network = InfiniteSquareNetwork(peps)
    return network, env
end

function ctmrg_iteration_benchmark(spec::CTMRGSpec, alg; T::Type = ComplexF64)
    network, env = setup_problem(spec; T)
    return @benchmarkable PEPSKit.ctmrg_iteration($network, $env, $alg)
end

function tomlify(spec::CTMRGSpec)
    return Dict(
        "unitcell" => collect(spec.unitcell),
        "Pspace" => tomlify(spec.Pspace),
        "Dspace" => tomlify(spec.Dspace),
        "chispace" => tomlify(spec.chispace),
    )
end

function untomlify(::Type{CTMRGSpec}, d)
    to_space = Base.Fix1(untomlify, VectorSpace)
    return CTMRGSpec(
        (Int(d["unitcell"][1]), Int(d["unitcell"][2])),
        to_space(d["Pspace"]),
        to_space(d["Dspace"]),
        to_space(d["chispace"]),
    )
end
