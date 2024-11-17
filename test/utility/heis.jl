module OpsHeis

export gen_gate, gen_siteop, gen_bondop
using TensorKit

"""
Create 1-site operators for Heisenberg model
"""
function gen_siteop(name::String)
    Pspace = ℂ^2
    if name == "Id"
        return id(Pspace)
    end
    op = TensorMap(zeros, Pspace, Pspace)
    if name == "Nud"
        block(op, Trivial())[:] = [1.0 0.0; 0.0 1.0]
    elseif name == "Sp"
        block(op, Trivial())[:] = [0.0 1.0; 0.0 0.0]
    elseif name == "Sm"
        block(op, Trivial())[:] = [0.0 0.0; 1.0 0.0]
    elseif name == "Sz"
        block(op, Trivial())[:] = [1.0 0.0; 0.0 -1.0] / 2
    elseif name == "Sx"
        block(op, Trivial())[:] = [0.0 1.0; 1.0 0.0] / 2
    elseif name == "iSy"
        block(op, Trivial())[:] = [0.0 1.0; -1.0 0.0] / 2
    else
        throw(ArgumentError("Invalid 1-site spin operator"))
    end
    return op
end

"""
Create 2-site operators for Heisenberg model
"""
function gen_bondop(name1::String, name2::String)
    op1 = gen_siteop(name1)
    op2 = gen_siteop(name2)
    op = op1 ⊗ op2
    return op
end

"""
Create nearest neighbor gate for Heisenberg model
"""
function gen_gate(J::Float64=1.0; dens_shift::Bool=false)
    heis =
        J * (
            (1 / 2) * gen_bondop("Sp", "Sm") +
            (1 / 2) * gen_bondop("Sm", "Sp") +
            gen_bondop("Sz", "Sz")
        )
    if dens_shift
        heis = heis - (J / 4) * gen_bondop("Nud", "Nud")
    end
    return heis
end

end

module RhoMeasureHeis

export measrho_all, cal_Esite

using TensorKit
using PEPSKit
using Statistics: mean
using ..OpsHeis

function cal_mags(rho1ss::Matrix{<:AbstractTensorMap})
    Pspace = codomain(rho1ss[1, 1])[1]'
    Sas = [gen_siteop(name) for name in ("Sx", "iSy", "Sz")]
    return [collect(meas_site(Sa, rho1) for rho1 in rho1ss) for Sa in Sas]
end

function cal_spincor(rho2ss::Matrix{<:AbstractTensorMap})
    SpSm = gen_bondop("Sp", "Sm")
    SzSz = gen_bondop("Sz", "Sz")
    return collect(meas_bond(SpSm, rho2) + meas_bond(SzSz, rho2) for rho2 in rho2ss)
end

function cal_Esite(rho2sss::Vector{<:Matrix{<:AbstractTensorMap}})
    N1, N2 = size(rho2sss[1])
    gate1 = gen_gate(; dens_shift=false)
    # 1st neighbor bond energy
    ebond1s = [collect(meas_bond(gate1, rho2) for rho2 in rho2sss[n]) for n in 1:2]
    esite = sum(sum(ebond1s)) / (N1 * N2)
    return esite, ebond1s
end

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
