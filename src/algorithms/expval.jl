function MPSKit.expectation_value(envs::CTMRGEnv,ham::AbstractTensorMap{S,1,1}) where S
    peps = envs.peps;
    result = Matrix{eltype(ham)}(undef,size(peps,1),size(peps,2));

    for r in 1:size(peps,1), c in 1:size(peps,2)
        e = @tensor envs.edges[WEST,r,c][1 2 3;4]*
            envs.corners[NORTHWEST,r,c][4;5]*
            envs.edges[NORTH,r,c][5 6 7;8]*
            envs.corners[NORTHEAST,r,c][8;9]*
            envs.edges[EAST,r,c][9 10 11;12]*
            envs.corners[SOUTHEAST,r,c][12;13]*
            envs.edges[SOUTH,r,c][13 14 15;16]*
            envs.corners[SOUTHWEST,r,c][16;1]*
            peps[r,c][17;6 10 14 2]*
            conj(peps[r,c][18;7 11 15 3])*
            ham[18;17]

        n = @tensor envs.edges[WEST,r,c][1 2 3;4]*
            envs.corners[NORTHWEST,r,c][4;5]*
            envs.edges[NORTH,r,c][5 6 7;8]*
            envs.corners[NORTHEAST,r,c][8;9]*
            envs.edges[EAST,r,c][9 10 11;12]*
            envs.corners[SOUTHEAST,r,c][12;13]*
            envs.edges[SOUTH,r,c][13 14 15;16]*
            envs.corners[SOUTHWEST,r,c][16;1]*
            peps[r,c][17;6 10 14 2]*
            conj(peps[r,c][17;7 11 15 3])

        @diffset result[r,c] = e/n;
    end
    
    result
end
