module TimeEvolBenchmarks

using BenchmarkTools
using TensorKit
using PEPSKit
using PEPSKit.MPSKitModels: hubbard_space

const SUITE = BenchmarkGroup()

"""
A simple update scenario on a 2 x 2 unit cell: the Hamiltonian `H` and the initial state `psi`,
evolved with bond dimension `D` (the truncation keeps the initial virtual spaces).
"""
struct SUSpec{H, S}
    H::H
    psi::S
    D::Int
    purified::Bool
end

function heisenberg_peps(D::Int; T = Float64)
    H = heisenberg_XYZ(InfiniteSquare(2, 2))
    psi = InfinitePEPS(randn, T, ℂ^2, ℂ^D; unitcell = (2, 2))
    return SUSpec(H, psi, D, true)
end

function heisenberg_u1_peps(n::Int; T = Float64)
    H = heisenberg_XXZ(T, U1Irrep, InfiniteSquare(2, 2))
    V = U1Space(0 => 2n, 1 // 2 => n, -1 // 2 => n)
    psi = InfinitePEPS(randn, T, physicalspace(H)[1, 1], V; unitcell = (2, 2))
    return SUSpec(H, psi, dim(V), true)
end

function hubbard_peps(n::Int; T = Float64)
    H = hubbard_model(T, Trivial, U1Irrep, InfiniteSquare(2, 2); t = 1.0, U = 6.0, mu = 3.0)
    V = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2n, (1, 1 // 2) => n, (1, -1 // 2) => n)
    psi = InfinitePEPS(randn, T, hubbard_space(Trivial, U1Irrep), V; unitcell = (2, 2))
    return SUSpec(H, psi, dim(V), true)
end

function heisenberg_pepo(D::Int; T = Float64)
    H = heisenberg_XYZ(InfiniteSquare(2, 2))
    psi = InfinitePEPO(randn, T, ℂ^2, ℂ^D; unitcell = (2, 2, 1))
    return SUSpec(H, psi, D, false)
end

"""
Benchmark one simple update iteration, i.e. the application of all Trotter gates of `spec.H`,
starting from weights obtained from a few warm-up iterations.
"""
function su_iter_benchmark(spec::SUSpec; dt = 1.0e-2, nwarmup = 3)
    alg = SimpleUpdate(; trunc = truncrank(spec.D), spec.purified)
    circuit = PEPSKit.trotterize(spec.H, dt)
    psi, env = spec.psi, SUWeight(spec.psi)
    for _ in 1:nwarmup
        psi, env, = PEPSKit.su_iter(psi, circuit, alg, env)
    end
    return @benchmarkable PEPSKit.su_iter($psi, $circuit, $alg, $env)
end

const SCENARIOS = [
    ("Heisenberg PEPS", "Trivial", "D8") => () -> heisenberg_peps(8),
    ("Heisenberg PEPS", "Trivial", "D16") => () -> heisenberg_peps(16),
    ("Heisenberg PEPS", "U1", "D8") => () -> heisenberg_u1_peps(2),
    ("Heisenberg PEPS", "U1", "D16") => () -> heisenberg_u1_peps(4),
    ("Hubbard PEPS", "fZ2xU1", "D8") => () -> hubbard_peps(2),
    ("Hubbard PEPS", "fZ2xU1", "D16") => () -> hubbard_peps(4),
    ("Heisenberg PEPO", "Trivial", "D4") => () -> heisenberg_pepo(4),
    ("Heisenberg PEPO", "Trivial", "D6") => () -> heisenberg_pepo(6),
]

for (groupkeys, spec) in SCENARIOS
    SUITE[("su_iter", groupkeys...)] = su_iter_benchmark(spec())
end

end # module
