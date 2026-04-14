using Test
using Random
using PEPSKit
using TensorKit
using TensorOperations: tensorcontract

using PEPSKit: eachcoordinate, _next_coordinate
using PEPSKit: EnlargedCorner, HalfInfiniteEnv, FullInfiniteEnv
using PEPSKit: half_infinite_environment, full_infinite_environment
using PEPSKit: simultaneous_projectors, contract_projectors
using PEPSKit: renormalize_northwest_corner, renormalize_northeast_corner,
    renormalize_southeast_corner, renormalize_southwest_corner
using PEPSKit: random_start_vector

# settings
Random.seed!(91283219347)
stype = ComplexF64

renormalize_corner_fns = (
    renormalize_northwest_corner, renormalize_northeast_corner,
    renormalize_southeast_corner, renormalize_southwest_corner,
)

function test_ctmrg_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )

    @testset "CTMRG PEPS contractions" begin
        peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
        env = CTMRGEnv(randn, stype, peps, chis_north, chis_east, chis_south, chis_west)

        n = InfiniteSquareNetwork(peps)

        test_contractions(n, env)
    end

    @testset "CTMRG PartitionFunction contractions" begin
        pf = InfinitePartitionFunction(randn, stype, Nspaces, Espaces)
        env = CTMRGEnv(randn, stype, pf, chis_north, chis_east, chis_south, chis_west)
        n = InfiniteSquareNetwork(pf)

        test_contractions(n, env)
    end

    @testset "CTMRG PEPO contractions" begin
        pepo = InfinitePEPO(randn, stype, Pspaces, Pspaces, Pspaces)
        peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
        n = InfiniteSquareNetwork(peps, pepo)
        env = CTMRGEnv(randn, stype, n, chis_north, chis_east, chis_south, chis_west)

        test_contractions(n, env)
    end

    return nothing
end

function test_contractions(n::InfiniteSquareNetwork, env::CTMRGEnv)
    dirs_and_coordinates = eachcoordinate(n, 1:4)

    # initialize dense and sparse enlarged corners
    sparse_enlarged_corners = map(dirs_and_coordinates) do co
        return EnlargedCorner(n, env, co)
    end
    dense_enlarged_corners = map(TensorMap, sparse_enlarged_corners)

    # initialize sparse and dense half-inifite environments
    sparse_halfinf_envs = map(dirs_and_coordinates) do co
        co´ = _next_coordinate(co, size(env)[2:3]...)
        return HalfInfiniteEnv(
            sparse_enlarged_corners[co...], sparse_enlarged_corners[co´...]
        )
    end
    dense_halfinf_envs = map(TensorMap, sparse_halfinf_envs)
    # also compute directly from dense enlarged corners, for consistency with current implementation
    dense_halfinf_envs_bis = map(dirs_and_coordinates) do co
        co´ = _next_coordinate(co, size(env)[2:3]...)
        return half_infinite_environment(
            dense_enlarged_corners[co...], dense_enlarged_corners[co´...]
        )
    end

    # initialize sparse and dense full-inifite environments
    sparse_fullinf_envs = map(dirs_and_coordinates) do co
        rowsize, colsize = size(env)[2:3]
        co2 = _next_coordinate(co, rowsize, colsize)
        co3 = _next_coordinate(co2, rowsize, colsize)
        co4 = _next_coordinate(co3, rowsize, colsize)
        return FullInfiniteEnv(
            sparse_enlarged_corners[co4...],
            sparse_enlarged_corners[co...],
            sparse_enlarged_corners[co2...],
            sparse_enlarged_corners[co3...],
        )
    end
    dense_fullinf_envs = map(TensorMap, sparse_fullinf_envs)
    # also compute directly from dense enlarged corners, for consistency with current implementation
    dense_fullinf_envs_bis = map(dirs_and_coordinates) do co
        rowsize, colsize = size(env)[2:3]
        co2 = _next_coordinate(co, rowsize, colsize)
        co3 = _next_coordinate(co2, rowsize, colsize)
        co4 = _next_coordinate(co3, rowsize, colsize)
        return full_infinite_environment(
            dense_enlarged_corners[co4...],
            dense_enlarged_corners[co...],
            dense_enlarged_corners[co2...],
            dense_enlarged_corners[co3...],
        )
    end

    # SVD half and full infinite environments
    (P_left_half, P_right_half), info_half = simultaneous_projectors(
        dense_enlarged_corners, env, HalfInfiniteProjector()
    )
    U_half, S_half, V_half = info_half.U, info_half.S, info_half.V
    (P_left_full, P_right_full), info_full = simultaneous_projectors(
        dense_enlarged_corners, env, FullInfiniteProjector()
    )
    U_full, S_full, V_full = info_full.U, info_full.S, info_full.V

    # check projector computation for both types of environments,
    # comparing dense and sparse implementations
    return foreach(dirs_and_coordinates) do co
        dir, r, c = co

        co2 = _next_coordinate(co, size(env)[2:3]...)
        co3 = _next_coordinate(co2, size(env)[2:3]...)
        co4 = _next_coordinate(co3, size(env)[2:3]...)

        ## HalfInfiniteEnv

        shenv = sparse_halfinf_envs[dir, r, c]
        dhenv = dense_halfinf_envs[dir, r, c]
        dhenv_bis = dense_halfinf_envs_bis[dir, r, c]
        @test dhenv ≈ dhenv_bis

        # application
        xr = random_start_vector(shenv)
        xl = randn(storagetype(shenv), codomain(shenv))
        @test shenv(xr, Val(false)) ≈ half_infinite_environment(dhenv, xr)
        @test shenv(xl, Val(true)) ≈ half_infinite_environment(xl, dhenv)

        # projector computation
        P_left_sparse, P_right_sparse = contract_projectors(
            U_half[dir, r, c], S_half[dir, r, c], V_half[dir, r, c], shenv
        )
        P_left_dense, P_right_dense = contract_projectors(
            U_half[dir, r, c], S_half[dir, r, c], V_half[dir, r, c],
            dense_enlarged_corners[co...], dense_enlarged_corners[co2...],
        )
        @test P_left_sparse ≈ P_left_dense
        @test P_right_sparse ≈ P_right_dense
        @test P_left_sparse ≈ P_left_half[dir, r, c]
        @test P_right_sparse ≈ P_right_half[dir, r, c]


        ## FullInfiniteEnv

        sfenv = sparse_fullinf_envs[dir, r, c]
        dfenv = dense_fullinf_envs[dir, r, c]
        dfenv_bis = dense_fullinf_envs_bis[dir, r, c]
        @test dfenv ≈ dfenv_bis

        # application
        xl = randn(storagetype(sfenv), codomain(sfenv))
        xr = random_start_vector(sfenv)
        @test sfenv(xr, Val(false)) ≈ full_infinite_environment(dfenv, xr)
        @test sfenv(xl, Val(true)) ≈ full_infinite_environment(xl, dfenv)

        # projector computation
        P_left_sparse, P_right_sparse = contract_projectors(
            U_full[dir, r, c], S_full[dir, r, c], V_full[dir, r, c], sfenv
        )
        P_left_dense, P_right_dense = contract_projectors(
            U_full[dir, r, c], S_full[dir, r, c], V_full[dir, r, c],
            half_infinite_environment(
                dense_enlarged_corners[co4...], dense_enlarged_corners[co...]
            ),
            half_infinite_environment(
                dense_enlarged_corners[co2...], dense_enlarged_corners[co3...]
            ),
        )
        @test P_left_sparse ≈ P_left_dense
        @test P_right_sparse ≈ P_right_dense
        @test P_left_sparse ≈ P_left_full[dir, r, c]
        @test P_right_sparse ≈ P_right_full[dir, r, c]
    end

    return foreach(dirs_and_coordinates) do co
        dir, r, c = co

        ## Corner renormalization

        C_sparse = renormalize_corner_fns[dir](
            (r, c), sparse_enlarged_corners, P_left, P_right
        )
        C_dense = renormalize_corner_fns[dir](
            (r, c), dense_enlarged_corners, P_left, P_right
        )
        @test C_sparse ≈ C_dense
    end
end

@testset "Random Cartesian spaces" begin
    unitcell = (3, 3)

    Pspaces = ComplexSpace.(rand(2:3, unitcell...))
    Nspaces = ComplexSpace.(rand(2:4, unitcell...))
    Espaces = ComplexSpace.(rand(2:4, unitcell...))
    chis_north = ComplexSpace.(rand(5:10, unitcell...))
    chis_east = ComplexSpace.(rand(5:10, unitcell...))
    chis_south = ComplexSpace.(rand(5:10, unitcell...))
    chis_west = ComplexSpace.(rand(5:10, unitcell...))

    test_ctmrg_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
end

@testset "Specific U1 spaces" begin
    unitcell = (2, 2)

    PA = U1Space(-1 => 1, 0 => 1)
    PB = U1Space(0 => 1, 1 => 1)
    Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
    Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)

    Pspaces = [PA PB; PB PA]
    Nspaces = [Vpeps Vpeps'; Vpeps' Vpeps]
    chis = [Venv Venv; Venv Venv]

    test_ctmrg_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)

    # 4x4 unit cell with all 32 inequivalent bonds
    #
    #    10     4     7    32
    #     |     |     |     |
    #  3--A--1--B--5--C--8--D--3
    #     |     |     |     |
    #     2     6     9    11
    #     |     |     |     |
    # 14--E-12--F-15--G-17--H-14
    #     |     |     |     |
    #    13    16    18    19
    #     |     |     |     |
    # 22--I-20--J-23--K-25--L-22
    #     |     |     |     |
    #    21    24    26    27
    #     |     |     |     |
    # 29--M-28--N-30--O-31--P-29
    #     |     |     |     |
    #    10     4     7    32

    phys_space = Vect[U1Irrep](1 => 1, -1 => 1)
    corner_space = Vect[U1Irrep](0 => 1, 1 => 1, -1 => 1)
    vspaces = map(i -> Vect[U1Irrep](0 => 1 + i % 4, 1 => i ÷ 4 % 4, -2 => i ÷ 16), 1:32)
    @test length(Set(vspaces)) == 32

    Espaces = [
        vspaces[1] vspaces[5] vspaces[8] vspaces[3]
        vspaces[12] vspaces[15] vspaces[17] vspaces[14]
        vspaces[20] vspaces[23] vspaces[25] vspaces[22]
        vspaces[28] vspaces[30] vspaces[31] vspaces[29]
    ]

    Nspaces = [
        vspaces[10] vspaces[4] vspaces[7] vspaces[32]
        vspaces[2] vspaces[6] vspaces[9] vspaces[11]
        vspaces[13] vspaces[16] vspaces[18] vspaces[19]
        vspaces[21] vspaces[24] vspaces[26] vspaces[27]
    ]
    Pspaces = fill(phys_space, (4, 4))
    chis = fill(corner_space, (4, 4))

    test_ctmrg_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
end

@testset "Random fermionic spaces" begin
    unitcell = (3, 3)

    S = Vect[FermionParity]
    pdims = rand(1:2, unitcell..., 2)
    vdims = rand(2:4, unitcell..., 2)
    edims = rand(3:6, unitcell..., 2)

    function _construct_space(ds::Array{<:Int, 3})
        V = map(Iterators.product(axes(ds)[1:2]...)) do (r, c)
            return S(0 => ds[r, c, 1], 1 => ds[r, c, 2])
        end
        return V
    end

    Pspaces = _construct_space(pdims)
    Nspaces = _construct_space(vdims)
    Espaces = _construct_space(vdims)
    chis_north = _construct_space(edims)
    chis_east = _construct_space(edims)
    chis_south = _construct_space(edims)
    chis_west = _construct_space(edims)

    test_ctmrg_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
end
