using Test
using Printf
using Random
using TensorKit
using PEPSKit
using LinearAlgebra
import Statistics: mean

module MeasureHeis

export cal_spincor, measure_heis

using TensorKit
import MPSKitModels: S_x, S_y, S_z, S_exchange
using PEPSKit

"""
Measure magnetization on each site
"""
function cal_mags(peps::InfinitePEPS, env::CTMRGEnv)
    Nr, Nc = size(peps)
    lattice = collect(space(t, 1) for t in peps.A)
    # detect symmetry on physical axis
    symm = sectortype(space(peps.A[1, 1]))
    if symm == Trivial
        Sas = real.([S_x(symm), im * S_y(symm), S_z(symm)])
    elseif symm == U1Irrep
        # only Sz preserves <Sz>
        Sas = real.([S_z(symm)])
    else
        throw(ArgumentError("Unrecognized symmetry on physical axis"))
    end
    return [
        collect(
            expectation_value(
                peps, LocalOperator(lattice, (CartesianIndex(r, c),) => Sa), env
            ) for (r, c) in Iterators.product(1:Nr, 1:Nc)
        ) for Sa in Sas
    ]
end

"""
Measure spin correlation ⟨Sᵢ⋅Sⱼ⟩ on each nearest neighbor bond
"""
function cal_spincor(peps::InfinitePEPS, env::CTMRGEnv)
    Nr, Nc = size(peps)
    symm = sectortype(peps.A[1, 1])
    op = real(S_exchange(ComplexF64, symm))
    lattice = collect(space(t, 1) for t in peps.A)
    corHs = collect(
        expectation_value(
            peps,
            LocalOperator(lattice, (CartesianIndex(r, c), CartesianIndex(r, c + 1)) => op),
            env,
        ) for (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    corVs = collect(
        expectation_value(
            peps,
            LocalOperator(lattice, (CartesianIndex(r, c), CartesianIndex(r - 1, c)) => op),
            env,
        ) for (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    return corHs, corVs
end

"""
Measure physical quantities for Heisenberg model
"""
function measure_heis(peps::InfinitePEPS, H::LocalOperator, env::CTMRGEnv)
    results = Dict{String,Any}()
    Nr, Nc = size(peps)
    results["e_site"] = cost_function(peps, env, H) / (Nr * Nc)
    results["mag"] = cal_mags(peps, env)
    results["mag_norm"] = collect(
        norm([mags[r, c] for mags in results["mag"]]) for
        (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    results["corH"], results["corV"] = cal_spincor(peps, env)
    return results
end

end

import .MeasureHeis: measure_heis
