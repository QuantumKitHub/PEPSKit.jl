# Shared helpers for the CTMRG gradient tests

# energy and gradient through the reverse rule of `leading_boundary`, starting from `env`
function energy_and_gradient(psi, env, ctmrg_alg, gradient_alg, H)
    E, gs = Zygote.withgradient(psi) do ψ
        env′, = PEPSKit.hook_pullback(leading_boundary, env, ψ, ctmrg_alg; alg_rrule = gradient_alg)
        return cost_function(ψ, env′, H)
    end
    return E, only(gs)
end

# relative distance of a gradient to the reference
gradient_error(g, gref) = norm(g - gref) / norm(gref)

# directional derivative of the energy along `dir` by a Richardson-extrapolated central
# difference, reconverging the environment at every point
function finite_difference_derivative(psi, env, dir, ctmrg_alg, H; h = 1.0e-4)
    function f(α)
        (ψ, e), = PEPSKit.peps_retract((psi, env), dir, α)
        e′, = leading_boundary(e, ψ, ctmrg_alg)
        return cost_function(ψ, e′, H)
    end
    d(h) = (f(h) - f(-h)) / (2h)
    return (4 * d(h) - d(2h)) / 3
end

# the same directional derivative from a gradient
function gradient_derivative(psi, env, dir, g)
    _, ξ = PEPSKit.peps_retract((psi, env), dir, 0)
    return PEPSKit.real_inner(nothing, ξ, g)
end
