module MeasureHeis

export gen_gate, measure_heis

using TensorKit
import MPSKitModels: S_x, S_y, S_z, S_exchange
using PEPSKit
using Statistics: mean

"""
Measure magnetization on each site
"""
function cal_mags(peps::InfinitePEPS, envs::CTMRGEnv)
    Nr, Nc = size(peps)
    lattice = collect(space(t, 1) for t in peps.A)
    Sas = real.([S_x(), im * S_y(), S_z()])
    return [
        collect(
            expectation_value(
                peps, LocalOperator(lattice, (CartesianIndex(r, c),) => Sa), envs
            ) for (r, c) in Iterators.product(1:Nr, 1:Nc)
        ) for Sa in Sas
    ]
end

"""
Measure physical quantities for Heisenberg model
"""
function measure_heis(peps::InfinitePEPS, H::LocalOperator, envs::CTMRGEnv)
    results = Dict{String,Any}()
    Nr, Nc = size(peps)
    results["e_site"] = costfun(peps, envs, H) / (Nr * Nc)
    results["mag"] = cal_mags(peps, envs)
    results["mag_norm"] = collect(
        norm([results["mag"][n][r, c] for n in 1:3]) for
        (r, c) in Iterators.product(1:Nr, 1:Nc)
    )
    return results
end

"""
Create nearest neighbor gate for Heisenberg model
"""
function gen_gate(J::Float64=1.0; dens_shift::Bool=false)
    heis = J * real(S_exchange())
    if dens_shift
        Pspace = ℂ^2
        heis = heis - (J / 4) * id(Pspace) ⊗ id(Pspace)
    end
    return heis
end

end
