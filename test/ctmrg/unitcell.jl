using Test
using Random
using PEPSKit
using PEPSKit: _prev, _next, ctmrg_iter
using TensorKit

# initialize (3, 3) PEPS with unique N and E bond spaces
Random.seed!(91283219347)
unitcell = (3, 3)
Pspace = ℂ^2
Nspace = ℂ^2
Espace = ℂ^3
stype = ComplexF64
peps = InfinitePEPS(randn, stype, Pspace, Nspace, Espace; unitcell)

# initialize (3, 3) environment with unique environment spaces
C_type = tensormaptype(spacetype(peps[1]), 1, 1, storagetype(peps[1]))
T_type = tensormaptype(spacetype(peps[1]), 3, 1, storagetype(peps[1]))
corners = Array{C_type}(undef, 4, unitcell...)
edges = Array{T_type}(undef, 4, unitcell...)
for r in 1:unitcell[1], c in 1:unitcell[2]
    corners[1, _prev(r, end), _prev(c, end)] = TensorMap(randn, stype, ℂ^5, ℂ^2)
    edges[1, _prev(r, end), c] = TensorMap(
        randn, stype, ℂ^2 * space(peps[r, c], 1 + 1)' * space(peps[r, c], 1 + 1), ℂ^2
    )
    corners[2, _prev(r, end), _next(c, end)] = TensorMap(randn, stype, ℂ^2, ℂ^3)
    edges[2, r, _next(c, end)] = TensorMap(
        randn, stype, ℂ^3 * space(peps[r, c], 2 + 1)' * space(peps[r, c], 2 + 1), ℂ^3
    )
    corners[3, _next(r, end), _next(c, end)] = TensorMap(randn, stype, ℂ^3, ℂ^4)
    edges[3, _next(r, end), c] = TensorMap(
        randn, stype, ℂ^4 * space(peps[r, c], 3 + 1)' * space(peps[r, c], 3 + 1), ℂ^4
    )
    corners[4, _next(r, end), _prev(c, end)] = TensorMap(randn, stype, ℂ^4, ℂ^5)
    edges[4, r, _prev(c, end)] = TensorMap(
        randn, stype, ℂ^5 * space(peps[r, c], 4 + 1)' * space(peps[r, c], 4 + 1), ℂ^5
    )
end
env = CTMRGEnv(corners, edges)

# apply one CTMRG iteration with fixeds
ctm_alg = CTMRG(; trscheme=FixedSpaceTruncation())
env′, = ctmrg_iter(peps, env, ctm_alg)

# compute random expecation value to test matching bonds
random_tensor = TensorMap(randn, scalartype(peps), Pspace, Pspace)
random_op = repeat(
    LocalOperator(fill(ℂ^2, 1, 1), (CartesianIndex(1, 1),) => random_tensor), unitcell...
)
@test expectation_value(peps, random_op, env) isa Number
@test expectation_value(peps, random_op, env′) isa Number
