# Tensor traces
# -------------

"""
    str(t)

Fermionic supertrace by using `@tensor`.
"""
str(t::AbstractTensorMap) = _str(BraidingStyle(sectortype(t)), t)
_str(::Bosonic, t::AbstractTensorMap) = tr(t)
@generated function _str(::Fermionic, t::AbstractTensorMap{<:Any, <:Any, N, N}) where {N}
    tex = tensorexpr(:t, ntuple(identity, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $tex))
end

"""
    trmul(H, ρ)

Compute `tr(H * ρ)` without forming `H * ρ`.
"""
@generated function trmul(
        H::AbstractTensorMap{<:Any, S, N, N}, ρ::AbstractTensorMap{<:Any, S, N, N}
    ) where {S, N}
    Hex = tensorexpr(:H, ntuple(identity, N), ntuple(i -> i + N, N))
    ρex = tensorexpr(:ρ, ntuple(i -> i + N, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $Hex * $ρex))
end
