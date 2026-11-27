using Test
using Random
using PEPSKit
using TensorKit
using Adapt

function examples_heisenberg_triangular(AT)
    return @testset "Triangular AF Heisenberg ($AT)" begin
        # References
        # ----------
        # Variational optimization of projected entangled-pair states on the triangular lattice
        # J. Naumann, J. Eisert, and P. Schmoll
        # Phys. Rev. B 113, 045117 (2026), arXiv:2510.04907
        # Table 1, native triangular CTMRG with a single-site spiral (120°) ansatz at D = 2
        E_ref = -0.54419

        # initialize parameters
        Dbond = 2
        χenv = 16
        Pspace = ComplexSpace(2)
        Vspace = ComplexSpace(Dbond)
        boundary_alg = (; alg = :SimultaneousCTMRGTriangular, tol = 1.0e-9, maxiter = 300, miniter = 4)
        gradient_alg = (; solver_alg = (; alg = :GMRES, krylovdim = 30), tol = 1.0e-7, maxiter = 10)
        optimizer_alg = (; tol = 1.0e-4, maxiter = 100)
        heisenberg(lattice) = adapt(AT, heisenberg_XYZ(lattice; Jx = 1.0, Jy = 1.0, Jz = 1.0))

        function optimize_energy_per_site(H, peps₀, nsites)
            env₀, = leading_boundary(
                CTMRGEnvTriangular(
                    randn, ComplexF64, Vspace ⊗ Vspace', ComplexSpace(χenv);
                    unitcell = size(peps₀), shift = peps₀.shift,
                ),
                peps₀; boundary_alg...,
            )
            peps, _, E, = fixedpoint(
                H, peps₀, env₀;
                boundary_alg, gradient_alg, optimizer_alg, precondition_alg = nothing,
            )
            return E / nsites, peps
        end

        # the 120° order fits the three-sublattice unit cell [A B C; C A B; B C A],
        # which is stored with only three tensors as a sheared (1, 3) unit cell
        Random.seed!(1)
        peps₀ = adapt(
            AT, InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace; unitcell = (1, 3), shift = 1)
        )
        e_three, peps_three = optimize_energy_per_site(
            heisenberg(InfiniteTriangularThreeSublattice()), peps₀, 3
        )
        @test size(peps_three) == (1, 3)
        @info "Three-sublattice energy per site = $e_three"
        @test e_three ≈ E_ref atol = 3.0e-3

        # the same 120° state with a single tensor, using a sublattice rotation of the
        # Hamiltonian (spiral ansatz with q = (-2π/3, 2π/3))
        Random.seed!(1)
        lattice_rot = InfiniteTriangular(1, 1; rotation = (-2π / 3, 2π / 3))
        peps₀ = adapt(AT, InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace))
        e_rot, peps_rot = optimize_energy_per_site(heisenberg(lattice_rot), peps₀, 1)
        @test size(peps_rot) == (1, 1)
        @info "Single-site rotated energy per site = $e_rot"
        @test e_rot ≈ E_ref atol = 3.0e-3
        @test e_rot ≈ e_three atol = 3.0e-3
        # the explicit rotated state is a three-sublattice state with the same energy
        peps_rot3 = rotate_sublattices(peps_rot, lattice_rot)
        @test size(peps_rot3) == (1, 3) && peps_rot3.shift == 1
        env_rot3, = leading_boundary(
            CTMRGEnvTriangular(randn, ComplexF64, Vspace ⊗ Vspace', ComplexSpace(χenv); unitcell = (1, 3), shift = 1),
            peps_rot3; boundary_alg...,
        )
        H_three = heisenberg(InfiniteTriangularThreeSublattice())
        @test real(expectation_value(peps_rot3, H_three, env_rot3)) / 3 ≈ e_rot atol = 1.0e-6

        # a 2 × 2 unit cell frustrates the 120° order
        Random.seed!(1)
        peps₀ = adapt(AT, InfinitePEPSTriangular(randn, ComplexF64, Pspace, Vspace; unitcell = (2, 2)))
        e_2x2, = optimize_energy_per_site(heisenberg(InfiniteTriangular(2, 2)), peps₀, 4)
        @info "2 × 2 energy per site = $e_2x2"
        @test e_three < e_2x2
    end
end
