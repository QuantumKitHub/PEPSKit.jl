# Plot timings for the `ctmrg` benchmark group.
#
# Usage:
#   julia --project=bench/benchmark bench/benchmark/scripts/plot_ctmrg.jl [results.json] [outdir]
#
# `results.json` is a file produced by `BenchmarkTools.save(path, results)`.
# Defaults to `bench/benchmark/data/results.json`. One PNG is written per
# scenario under `outdir` (default: same directory as the input). The x-axis is
# the minimal bond dimension D parsed from the benchname (`D<D>_chi<χ>`); each
# (symmetry, χ) pair is a separate series. Multiple markers in a series at the
# same D come from multiple algorithms in the scenario.

using BenchmarkTools
using CairoMakie

const DEFAULT_INPUT = normpath(joinpath(@__DIR__, "..", "data", "results.json"))
const BENCHNAME_RE = r"^D(\d+)_chi(\d+)$"

function load_ctmrg(path::AbstractString)
    isfile(path) || error("results file not found: $(abspath(path))")
    results = BenchmarkTools.load(path)[1]
    haskey(results, "ctmrg") || error("no `ctmrg` group in $(path)")
    return results["ctmrg"]
end

function parse_benchname(bn::AbstractString)
    m = match(BENCHNAME_RE, bn)
    isnothing(m) && return nothing
    return (D = parse(Int, m[1]), chi = parse(Int, m[2]))
end

function plot_one(g_scen::BenchmarkGroup, scenario::AbstractString; outdir::AbstractString)
    # (symmetry, χ) -> Vector{(D, time_ms)} over all algorithms and unit cells.
    series = Dict{Tuple{String, Int}, Vector{Tuple{Int, Float64}}}()

    for alg_name in keys(g_scen)
        for sym_name in keys(g_scen[alg_name])
            sym_str = String(sym_name)
            for (bn, trial) in g_scen[alg_name][sym_name]
                meta = parse_benchname(String(bn))
                isnothing(meta) && continue
                key = (sym_str, meta.chi)
                t_ms = minimum(trial).time / 1e6
                pts = get!(series, key, Tuple{Int, Float64}[])
                push!(pts, (meta.D, t_ms))
            end
        end
    end
    isempty(series) && return nothing

    sorted_keys = sort!(collect(keys(series)))
    all_D = sort!(collect(unique(p[1] for pts in values(series) for p in pts)))
    colors = Makie.wong_colors()

    fig = Figure(; size = (700, 480))
    ax = Axis(
        fig[1, 1];
        title = "ctmrg / $(scenario)",
        xlabel = "minimal D",
        ylabel = "minimum time (ms)",
        yscale = log10,
        xticks = (all_D, string.(all_D)),
    )

    for (i, key) in enumerate(sorted_keys)
        sym, chi = key
        pts = sort(series[key]; by = first)
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        scatterlines!(ax, xs, ys;
            color = colors[mod1(i, length(colors))],
            label = "$sym, χ=$chi",
            markersize = 12,
        )
    end
    Legend(fig[1, 2], ax)

    mkpath(outdir)
    out = joinpath(outdir, "ctmrg_$(scenario).png")
    save(out, fig)
    println("wrote $(out)")
    return out
end

function (@main)(args::AbstractVector{<:AbstractString})
    input = isempty(args) ? DEFAULT_INPUT : args[1]
    outdir = length(args) >= 2 ? args[2] : dirname(abspath(input))
    g_ctmrg = load_ctmrg(input)
    for scenario in sort!(String.(collect(keys(g_ctmrg))))
        plot_one(g_ctmrg[scenario], scenario; outdir)
    end
end
