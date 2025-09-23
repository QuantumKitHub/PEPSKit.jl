using TensorKit
using PEPSKit
using MPSKit: add_physical_charge
using MPSKitModels: a_number, nꜛnꜜ, contract_onesite
using Test

vds = (ℂ^2, Rep[U₁](1 => 1, -1 => 1), Rep[SU₂](1 / 2 => 1))
@testset "LocalOperator $vd" for vd in vds
    t = randn(ComplexF64, vd ⊗ vd ← vd ⊗ vd)
    physical_spaces = fill(vd, (2, 2))

    terms = ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => t,)
    op = LocalOperator(physical_spaces, terms...)

    @test op isa LocalOperator
    @test length(op.terms) == 1
    @test sectortype(op) === sectortype(vd)
    @test spacetype(op) === typeof(vd)
    @test physicalspace(op) == physical_spaces

    @test real(last(only(real(op).terms))) == real(t)
    @test real(last(only(imag(op).terms))) == imag(t)

    op2 = 2 * op
    @test op2 isa LocalOperator
    @test typeof(op2) === typeof(op)
    @test physicalspace(op2) == physical_spaces

    @test real(last(only(real(op2).terms))) ≈ 2 * real(t)
    @test real(last(only(imag(op2).terms))) ≈ 2 * imag(t)

    op3 = op / 3
    @test op3 isa LocalOperator
    @test typeof(op3) === typeof(op)
    @test physicalspace(op3) == physical_spaces

    @test real(last(only(real(op3).terms))) ≈ real(t) / 3
    @test real(last(only(imag(op3).terms))) ≈ imag(t) / 3

    t2 = randn(vd ⊗ vd ← vd ⊗ vd)
    terms2 = ((CartesianIndex(2, 1), CartesianIndex(1, 2)) => t2,)
    op4 = LocalOperator(physical_spaces, terms2...)
    op5 = op + op4
    @test op5 isa LocalOperator
    @test physicalspace(op5) == physical_spaces
    @test length(op5.terms) == 2
end

@testset "Charge shifting" begin
    lattice = InfiniteSquare(1, 1)
    elt = ComplexF64
    U = 30.0

    # bosonic case
    cutoff = 2
    N = a_number(elt, U1Irrep; cutoff)
    H_U = U / 2 * contract_onesite(N, N - id(domain(N)))
    spaces = fill(space(H_U, 1), (lattice.Nrows, lattice.Ncols))
    H = LocalOperator(spaces, ((1, 1),) => H_U)
    tr_before = tr(last(only(H.terms)))
    # shift to unit filling
    caux = U1Irrep(1)
    H_shifted = add_physical_charge(H, fill(caux, size(H.lattice)...))
    # check if spaces were correctly shifted
    @test H_shifted.lattice == map(
        fuse, H.lattice, fill(U1Space(caux => 1)', size(H.lattice)...)
    )
    # check if trace is properly preserved
    tr_after = tr(last(only(H_shifted.terms)))
    @test abs(tr_before - tr_after) / abs(tr_before) < 1.0e-12

    # fermionic case
    symmetry = FermionParity ⊠ U1Irrep
    H_U = U * nꜛnꜜ(elt, U1Irrep, Trivial)
    spaces = fill(space(H_U, 1), (lattice.Nrows, lattice.Ncols))
    H = LocalOperator(spaces, ((1, 1),) => H_U)
    tr_before = tr(last(only(H.terms)))
    # shift to unit filling
    caux = symmetry((1, 1))
    H_shifted = add_physical_charge(H, fill(caux, size(H.lattice)...))
    # check if spaces were correctly shifted
    @test H_shifted.lattice == map(
        fuse, H.lattice, fill(Vect[symmetry](caux => 1)', size(H.lattice)...)
    )
    # check if trace is properly preserved
    tr_after = tr(last(only(H_shifted.terms)))
    @test abs(tr_before - tr_after) / abs(tr_before) < 1.0e-12
end
