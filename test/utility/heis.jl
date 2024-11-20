module RhoMeasureHeis

export gen_gate, measrho_all, cal_Esite

using TensorKit, PEPSKit, MPSKitModels
using Statistics: mean

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

"""
Measure magnetization on each site
"""
function cal_mags(rho1ss::Matrix{<:AbstractTensorMap})
    Sas = real.([S_x(), im * S_y(), S_z()])
    return [collect(meas_site(Sa, rho1) for rho1 in rho1ss) for Sa in Sas]
end

"""
Measure spin correlation on each nearest neighbor bond
"""
function cal_spincor(rho2ss::Matrix{<:AbstractTensorMap})
    SpSm = real(S_plus() ⊗ S_min())
    SzSz = real(S_z() ⊗ S_z())
    return collect(meas_bond(SpSm, rho2) + meas_bond(SzSz, rho2) for rho2 in rho2ss)
end

"""
Measure energy on each nearest neighbor bond
"""
function cal_Esite(rho2sss::Vector{<:Matrix{<:AbstractTensorMap}})
    N1, N2 = size(rho2sss[1])
    gate1 = gen_gate(; dens_shift=false)
    # 1st neighbor bond energy
    ebond1s = [collect(meas_bond(gate1, rho2) for rho2 in rho2sss[n]) for n in 1:2]
    esite = sum(sum(ebond1s)) / (N1 * N2)
    return esite, ebond1s
end

"""
Measure physical quantities for Heisenberg model
"""
function measrho_all(
    rho1ss::Matrix{<:AbstractTensorMap}, rho2sss::Vector{<:Matrix{<:AbstractTensorMap}}
)
    results = Dict{String,Any}()
    N1, N2 = size(rho1ss)
    results["e_site"], results["energy"] = cal_Esite(rho2sss)
    results["mag"] = cal_mags(rho1ss)
    results["mag_norm"] = collect(
        norm([results["mag"][n][r, c] for n in 1:3]) for
        (r, c) in Iterators.product(1:N1, 1:N2)
    )
    results["spincor"] = [cal_spincor(rho2ss) for rho2ss in rho2sss]
    return results
end

end
