using TensorKit
using PEPSKit
using PEPSKit: siterotl90, siterotr90, siterot180
using MPSKit: add_physical_charge
using TensorKitTensors.BosonOperators: b_num
using TensorKitTensors.HubbardOperators: ud_num
using Test

vds = (ℂ^2, Rep[U₁](1 => 1, -1 => 1), Rep[SU₂](1 / 2 => 1))

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    @testset "LocalOperator $vd" for vd in vds
        t = randn(ComplexF64, vd ⊗ vd ← vd ⊗ vd)
        physical_spaces = fill(vd, (2, 2))

        terms = ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => t,)
        op = LocalOperator(physical_spaces, terms...)
        @test TensorKit.storagetype(op) == TensorKit.storagetype(t)
        @test TensorKit.storagetype(op + LocalOperator(physical_spaces, ((2, 1), (2, 2)) => real(t))) == TensorKit.storagetype(t)

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
        N = b_num(elt, U1Irrep; cutoff)
        H_U = U / 2 * N * (N - id(domain(N)))
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
        H_U = U * ud_num(elt, U1Irrep, Trivial)
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

    # Cover bosonic, fermionic, combined, and non-Abelian symmetries.
    @testset "Factored charge shifting ($(sectortype(p)))" for (p, charges) in (
            (Vect[U1Irrep](-1 => 1, 0 => 1, 1 => 1), (1, -1, 2)),
            (Vect[FermionParity](0 => 1, 1 => 1), (1, 0, 1)),
            (Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1), ((1, 1), (1, 0), (0, -1))),
            (Vect[SU2Irrep](1 / 2 => 1), (1 / 2, 0, 1)),
        )
        spaces = fill(p, 1, 3)
        charge_lattice = reshape(sectortype(p).(collect(charges)), 1, 3)
        allsites = CartesianIndex.([(1, 3), (1, 1), (1, 2)])
        # Exercise one-site terms, MPO endpoints, and an interior MPO factor.
        @testset "$n sites" for n in 1:3
            # Unsorted sites check that each factor receives its site's charge.
            sites = allsites[1:n]
            product = [randn(ComplexF64, p ← p) for _ in 1:n]
            gate = randn(ComplexF64, p^n ← p^n)
            mpo = n == 1 ? [gate] : PEPSKit.gate_to_mpo(gate; trunc = notrunc())
            @testset "$name" for (name, dense, factors) in (
                    ("Tensor product", foldl(⊗, product), product), ("MPO", gate, mpo),
                )
                H = LocalOperator(spaces, sites => factors)
                snapshot = deepcopy(H)
                shifted = add_physical_charge(H, charge_lattice)
                expected = add_physical_charge(LocalOperator(spaces, sites => dense), charge_lattice)
                shifted_sites, ws = only(shifted.terms)
                actual = if ws isa PEPSKit.TensorProductTerm
                    foldl(⊗, ws)
                elseif n == 2
                    @tensor O[a b; c d] := ws[1][a; c x] * ws[2][x b; d]
                else
                    @tensor O[a b c; d e f] := ws[1][a; d x] * ws[2][x b; e y] * ws[3][y c; f]
                end
                reconstructed = LocalOperator(physicalspace(shifted), shifted_sites => actual)
                # Physical spaces must match the dense charge shift.
                @test physicalspace(shifted) == physicalspace(expected)
                # Contracted factors must reproduce the dense result, including fermionic signs.
                @test only(values(reconstructed.terms)) ≈ only(values(expected.terms))
                # Charge shifting must leave the input operator unchanged.
                @test H == snapshot
                if n > 1 && !(ws isa PEPSKit.TensorProductTerm)
                    # Preserve virtual bond spaces on both sides of each MPO bond.
                    @test all(i -> domain(ws[i])[2] == domain(factors[i])[2], 1:(n - 1))
                    @test all(i -> codomain(ws[i])[1] == codomain(factors[i])[1], 2:n)
                end
            end
        end
    end

    unitcells = [(1, 1), (2, 3), (3, 3), (4, 3)]
    @testset "Site rotations on $uc unitcell" for uc in unitcells
        unrotated_inds = collect(CartesianIndices(uc))
        # use reverse(uc) to account for transposing when using rotl90, rotr90 on non-square unit cells
        rr_rotated_inds = rotr90(siterotr90.(collect(CartesianIndices(reverse(uc))), Ref(reverse(uc))))
        ll_rotated_inds = rotl90(siterotl90.(collect(CartesianIndices(reverse(uc))), Ref(reverse(uc))))
        half_rotated_inds = rot180(siterot180.(collect(CartesianIndices(uc)), Ref(uc)))

        @test unrotated_inds == rr_rotated_inds
        @test unrotated_inds == ll_rotated_inds
        @test unrotated_inds == half_rotated_inds
    end

    op_1x1 = LocalOperator([ℂ^2;;], ((1, 1), (1, 2)) => randn(ℂ^2, ℂ^2) ⊗ randn(ℂ^2, ℂ^2))
    # J1-J2 only has spin U(1) symmetry without sub-lattice rotation
    # See https://github.com/QuantumKitHub/MPSKitModels.jl/issues/57
    op_2x2 = add_physical_charge(
        j1_j2_model(ComplexF64, U1Irrep, InfiniteSquare(2, 2); sublattice = false),
        [
            U1Irrep(-1 // 2) U1Irrep(1 // 2)
            U1Irrep(1 // 2) U1Irrep(-1 // 2)
        ] # staggered charges to create non-uniform physical spaces
    )
    op_2x3 = LocalOperator(
        [
            ℂ^1 ℂ^2 ℂ^3
            ℂ^4 ℂ^5 ℂ^6
        ],

        (
            ((1, 1), (1, 2)) => randn(ℂ^1, ℂ^1) ⊗ randn(ℂ^2, ℂ^2),
            ((2, 1), (1, 1)) => randn(ℂ^4, ℂ^4) ⊗ randn(ℂ^1, ℂ^1),
            ((1, 2), (2, 3)) => randn(ℂ^2, ℂ^2) ⊗ randn(ℂ^6, ℂ^6),
            ((1, 3), (2, 2)) => randn(ℂ^3, ℂ^3) ⊗ randn(ℂ^5, ℂ^5),
        )...
    )
    operators = [op_1x1, op_2x2, op_2x3]
    @testset "Operator rotations on $(size(op)) operator" for op in operators
        @test rot180(rot180(op)) == op
        @test rotl90(rotl90(op)) == rot180(op) == rotr90(rotr90(op))
        @test physicalspace(rotl90(op)) == rotl90(physicalspace(op))
        @test physicalspace(rotr90(op)) == rotr90(physicalspace(op))
        @test physicalspace(rot180(op)) == rot180(physicalspace(op))
    end
end
