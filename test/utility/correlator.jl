using Test
using Random
using TensorKit
using PEPSKit
import MPSKitModels: TJOperators as tJ

Pspace = tJ.tj_space(Trivial, Trivial)
Vspace = Vect[FermionParity](0 => 2, 1 => 2)
Espace = Vect[FermionParity](0 => 3, 1 => 3)
Random.seed!(100)
peps = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
env = CTMRGEnv(rand, ComplexF64, peps, Espace);
lattice = collect(space(t, 1) for t in peps.A)

site0 = CartesianIndex(1, 1)
maxsep = 8
site1s = collect(site0 + CartesianIndex(0, i) for i in 2:2:maxsep)

op = tJ.S_exchange(ComplexF64, Trivial, Trivial);

vals1 = correlator(peps, op, site0, site1s, env)
vals2 = collect(begin
    O = LocalOperator(lattice, (site0, site1) => op)
    val = expectation_value(peps, O, env)
end for site1 in site1s)
@test vals1 ≈ vals2
