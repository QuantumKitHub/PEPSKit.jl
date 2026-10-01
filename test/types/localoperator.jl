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

    @testset "LocalOperator MPO bookkeeping" begin
        d = ℂ^2
        lattice = fill(d, 2, 2)
        dense = randn(Float64, d ⊗ d ← d ⊗ d)
        mpo = PEPSKit.gate_to_mpo(dense; trunc = notrunc())
        sites = CartesianIndex.([(3, 4), (3, 3)])
        original_sites, original_mpo = copy(sites), copy(mpo)
        operator = LocalOperator(lattice, sites => mpo)
        stored_sites, stored_mpo = only(operator.terms)

        # Construction translates a copy of the coordinates and must retain the caller's MPO order.
        # Mutating either input container afterwards must not alter the stored term.
        @test sites == original_sites
        reverse!(sites)
        reverse!(mpo)
        @test stored_sites == CartesianIndex.([(1, 2), (1, 1)])
        @test stored_mpo == original_mpo

        # Scaling every factor would multiply the operator by α^N; only one factor may change.
        # A complex coefficient must also widen the container without mutating the original tensors.
        α = 2 + 3im
        snapshot = deepcopy(stored_mpo)
        scaled_mpo = last(only((α * operator).terms))
        @test first(scaled_mpo) ≈ α * first(stored_mpo)
        @test last(scaled_mpo) === last(stored_mpo)
        @test stored_mpo == snapshot
        onsite = randn(ComplexF64, d ← d)
        mixed = operator + LocalOperator(lattice, ((2, 1),) => onsite)
        @test scalartype(mixed) == ComplexF64

        # Taking real/imaginary parts factorwise does not give the real/imaginary part of an MPO.
        @test_throws ArgumentError real(mixed)
        @test_throws ArgumentError imag(mixed)

        # Insertion and operator addition use different accumulation paths; both must reject MPO sums.
        # Dense terms sharing the same sites must still accumulate normally.
        inds = CartesianIndex.([(1, 1), (1, 2)])
        dense_operator = LocalOperator(lattice, inds => dense)
        mpo_operator = LocalOperator(lattice, inds => original_mpo)
        @test_throws ArgumentError PEPSKit.add_term!(mpo_operator, inds, dense)
        @test_throws ArgumentError PEPSKit.add_term!(dense_operator, inds, original_mpo)
        @test_throws ArgumentError mpo_operator + mpo_operator
        @test_throws ArgumentError mpo_operator + dense_operator
        @test_throws ArgumentError dense_operator + mpo_operator
        @test last(only((dense_operator + dense_operator).terms)) ≈ 2 * dense

        # Drop zero factors before they reach boundary-MPS normalization, where they could cause 0/0.
        @test isempty(LocalOperator(lattice, inds => [zero(first(original_mpo)), last(original_mpo)]).terms)

        # Unequal physical spaces expose accidental factor reordering during coordinate transformations.
        lattice = [ℂ^2 ℂ^3 ℂ^4; ℂ^5 ℂ^6 ℂ^7]
        dense = randn(Float64, ℂ^6 ⊗ ℂ^2 ← ℂ^6 ⊗ ℂ^2)
        mpo = PEPSKit.gate_to_mpo(dense; trunc = notrunc())
        operator = LocalOperator(lattice, ((2, 2), (1, 1)) => mpo)
        @test rotr90(rotl90(operator)) == operator
        @test last(only(rotr90(operator).terms)) == mpo
        sites, term = only(operator.terms)
        @test repeat(operator, 2, 1).terms == Dict(sites => term, (sites .+ CartesianIndex(2, 0)) => term)
    end

    @testset "MPO validation" begin
        d, b1, b2 = ℂ^2, ℂ^3, ℂ^4
        lattice = fill(d, 2, 2)
        first_tensor = randn(Float64, d ← d ⊗ b1)
        last_tensor = randn(Float64, b1 ⊗ d ← d)
        mpo = [first_tensor, last_tensor]
        sites = ((1, 1), (1, 2))

        # Reject malformed chains at construction, before any routing or contraction is attempted.
        @test_throws ArgumentError LocalOperator(lattice, CartesianIndex{2}[] => AbstractTensorMap[])
        @test_throws ArgumentError LocalOperator(lattice, ((1, 1),) => mpo)
        @test_throws ArgumentError LocalOperator(lattice, ((1, 1), (1, 1)) => mpo)
        @test_throws ArgumentError LocalOperator(lattice, sites => reverse(mpo))

        # Bond matching and the two physical legs are independent constraints.
        wrong_bond = randn(Float64, b2 ⊗ d ← d)
        wrong_input = randn(Float64, b1 ⊗ d ← ℂ^3)
        wrong_output = randn(Float64, b1 ⊗ ℂ^3 ← d)
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_bond])
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_input])
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_output])
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
