function MPSKit.expectation_value(state::InfPEPS,nn::NN,pars::InfNNHamChannels = params(state,nn))
    man  = pars.envm;

    tot = 0.0+0im

    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))

        tot += @tensor man.fp1RL[North,i,j][1,2,3,4]*
            man.AR[East,i,j][4,5,6,7]*
            man.AR[East,i+1,j][7,8,9,10]*
            man.fp1LR[South,i+1,j][10,11,12,13]*
            man.AL[West,i+1,j][13,14,15,16]*
            man.AL[West,i,j][16,17,18,1]*
            man.peps[i,j][17,19,5,2,20]*
            conj(man.peps[i,j][18,21,6,3,22])*
            man.peps[i+1,j][14,11,8,19,23]*
            conj(man.peps[i+1,j][15,12,9,21,24])*
            nn[20,22,23,24]

        tot += @tensor man.fp1RL[West,i,j][1,2,3,4]*
            man.AR[North,i,j][4,5,6,7]*
            man.AR[North,i,j+1][7,8,9,10]*
            man.fp1LR[East,i,j+1][10,11,12,13]*
            man.AL[South,i,j+1][13,14,15,16]*
            man.AL[South,i,j][16,17,18,1]*
            man.peps[i,j][2,17,19,5,20]*
            conj(man.peps[i,j][3,18,21,6,22])*
            man.peps[i,j+1][19,14,11,8,23]*
            conj(man.peps[i,j+1][21,15,12,9,24])*
            nn[20,22,23,24]
    end

    tot
end

function MPSKit.expectation_value(state::FinPEPS,nn::NN,pars::FinNNHamChannels = params(state,nn))
    #=
    contrast it with the infpeps code. We only had to add bound checks and normalization (ipeps is normalized in place)
    =#
    man  = pars.envm;

    tot = 0.0+0im
    normalization = 0.0+0im;
    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))
        if i < size(man.peps,1)
            tot += @tensor man.fp1RL[North,i,j][1,2,3,4]*
                man.AR[East,i,j][4,5,6,7]*
                man.AR[East,i+1,j][7,8,9,10]*
                man.fp1LR[South,i+1,j][10,11,12,13]*
                man.AL[West,i+1,j][13,14,15,16]*
                man.AL[West,i,j][16,17,18,1]*
                man.peps[i,j][17,19,5,2,20]*
                conj(man.peps[i,j][18,21,6,3,22])*
                man.peps[i+1,j][14,11,8,19,23]*
                conj(man.peps[i+1,j][15,12,9,21,24])*
                nn[20,22,23,24]

            normalization += @tensor man.fp1RL[North,i,j][1,2,3,4]*
                man.AR[East,i,j][4,5,6,7]*
                man.AR[East,i+1,j][7,8,9,10]*
                man.fp1LR[South,i+1,j][10,11,12,13]*
                man.AL[West,i+1,j][13,14,15,16]*
                man.AL[West,i,j][16,17,18,1]*
                man.peps[i,j][17,19,5,2,20]*
                conj(man.peps[i,j][18,21,6,3,20])*
                man.peps[i+1,j][14,11,8,19,23]*
                conj(man.peps[i+1,j][15,12,9,21,23])
        end

        if j < size(man.peps,2)
            tot += @tensor man.fp1RL[West,i,j][1,2,3,4]*
                man.AR[North,i,j][4,5,6,7]*
                man.AR[North,i,j+1][7,8,9,10]*
                man.fp1LR[East,i,j+1][10,11,12,13]*
                man.AL[South,i,j+1][13,14,15,16]*
                man.AL[South,i,j][16,17,18,1]*
                man.peps[i,j][2,17,19,5,20]*
                conj(man.peps[i,j][3,18,21,6,22])*
                man.peps[i,j+1][19,14,11,8,23]*
                conj(man.peps[i,j+1][21,15,12,9,24])*
                nn[20,22,23,24]

            normalization += @tensor man.fp1RL[West,i,j][1,2,3,4]*
                man.AR[North,i,j][4,5,6,7]*
                man.AR[North,i,j+1][7,8,9,10]*
                man.fp1LR[East,i,j+1][10,11,12,13]*
                man.AL[South,i,j+1][13,14,15,16]*
                man.AL[South,i,j][16,17,18,1]*
                man.peps[i,j][2,17,19,5,20]*
                conj(man.peps[i,j][3,18,21,6,20])*
                man.peps[i,j+1][19,14,11,8,23]*
                conj(man.peps[i,j+1][21,15,12,9,23])
        end
    end

    tot/normalization
end
