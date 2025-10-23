@doc """
    time_evolve(ψ₀, H, dt, nstep, alg, envs₀; kwargs...) -> (ψ, envs, info)

Time-evolve the initial state `ψ₀` with Hamiltonian `H` over 
the time span `dt * nstep` based on Trotter decomposition. 

## Arguments

- `ψ₀::Union{InfinitePEPS, InfinitePEPO}`: Initial state.
- `H::LocalOperator`: Hamiltonian operator (time-independent).
- `dt::Float64`: Trotter time evolution step.
- `nstep::Int`: Number of Trotter steps to be taken, such that the evolved time span is `dt * nstep`.
- `alg`: Time evolution algorithm.
- `envs₀`: Environment of the initial state.

## Keyword Arguments

- `imaginary_time::Bool=false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful for using this
    function to compute the ground state of a Hamiltonian, or to compute finite-temperature
    properties of a system.
- `tol::Float64 = 0.0`: Tolerance of the change in SUWeight (for simple update; default 1.0e-9) 
    or energy (for full update; default 1.0e-8) to determine if the ground state search has converged.
"""
time_evolve
