using PEPSKit,MPSKit,TensorKit,Test,TestExtras

println("------------------------------------")
println("|     States                       |")
println("------------------------------------")
@timedtestset "($T,$D,$d,$elt)" for (T,D,d,elt) in [
        (InfPEPS,ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (InfPEPS,ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32),
        (FinPEPS,ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (FinPEPS,ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
        ]

    nrows = rand(1:10);
    ncols = rand(1:10);
    st = T(map(Iterators.product(1:nrows,1:ncols)) do (i,j)
        TensorMap(rand,elt,D*D*D'*D',d')
    end)

    #does rotation work
    @test rotl90(st[1,1]) ≈ rotl90(st)[end,1];
    @test rotr90(st[1,1]) ≈ rotr90(st)[1,end];

    @test size(st)[1] ≈ size(rotl90(st))[2];
    @test size(st)[2] ≈ size(rotl90(st))[1];

    for d in Dirs
        for (i,j) in Iterators.product(1:nrows,1:ncols)
            or_tens = st[i,j];
            (ti,tj) = rotate_north((i,j),(nrows,ncols),d);
            ne_tens = rotate_north(st,d)[ti,tj];

            @test norm(or_tens) ≈ norm(ne_tens);
            @tensor rotate_north(or_tens,d) ≈ ne_tens;
        end
    end

    #does copying work
    flatcopy_st = copy(st);
    deepcopy_st = deepcopy(st);
    for (i,j) in Iterators.product(1:nrows,1:ncols)
        @test st[i,j] === flatcopy_st[i,j];
        flatcopy_st[i,j] = similar(flatcopy_st[i,j]);
        @test !(st[i,j] === flatcopy_st[i,j]);

        @test !(st[i,j] === deepcopy_st[i,j]);
    end
end

@timedtestset "window peps ($D,$d,$elt)" for (D,d,elt) in [
        (ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1),ComplexF32)
        ]

    nrows_env = rand(1:10);ncols_env = rand(1:10);
    outside = InfPEPS(map(Iterators.product(1:nrows_env,1:ncols_env)) do (i,j)
        TensorMap(rand,elt,D*D*D'*D',d')
    end)

    r_unit = rand(1:5); c_unit = rand(1:5);
    nrows = r_unit*nrows_env;
    ncols = c_unit*ncols_env;

    inside = FinPEPS(map(Iterators.product(1:nrows,1:ncols)) do (i,j)
        TensorMap(rand,elt,D*D*D'*D',d')
    end)

    st = WinPEPS(inside,outside);

    #does rotation work
    @test rotl90(st[1,1]) ≈ rotl90(st)[end,1];
    @test rotr90(st[1,1]) ≈ rotr90(st)[1,end];

    @test size(st)[1] ≈ size(rotl90(st))[2];
    @test size(st)[2] ≈ size(rotl90(st))[1];

    for d in Dirs
        for (i,j) in Iterators.product(1:nrows,1:ncols)
            or_tens = st[i,j];
            (ti,tj) = rotate_north((i,j),(nrows,ncols),d);
            ne_tens = rotate_north(st,d)[ti,tj];

            @test norm(or_tens) ≈ norm(ne_tens);
            @tensor rotate_north(or_tens,d) ≈ ne_tens;
        end
    end

    #does copying work
    flatcopy_st = copy(st);
    deepcopy_st = deepcopy(st);
    for (i,j) in Iterators.product(1:nrows,1:ncols)
        @test st[i,j] === flatcopy_st[i,j];
        flatcopy_st[i,j] = similar(flatcopy_st[i,j]);
        @test !(st[i,j] === flatcopy_st[i,j]);

        @test !(st[i,j] === deepcopy_st[i,j]);
    end
end

println("------------------------------------")
println("|     Environments                 |")
println("------------------------------------")
