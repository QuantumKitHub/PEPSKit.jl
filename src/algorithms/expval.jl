MPSKit.expectation_value(man::Union{InfNNHamChannels,FinNNHamChannels,WinNNHamChannels,FinNNHamCors},nn::NN) = expectation_value(man.envm,nn);
MPSKit.expectation_value(man::Union{InfNNHamChannels,FinNNHamChannels,WinNNHamChannels,FinNNHamCors},opp::MPSKit.MPSBondTensor) = expectation_value(man.envm,opp);

function MPSKit.expectation_value(man::Union{InfEnvManager,WinEnvManager,FinEnvManager},opp::MPSKit.MPSBondTensor)
    expval = map(Iterators.product(1:size(man.peps,1),1:size(man.peps,2))) do (i,j)
        e = @tensor fp1LR(man,North,i,j)[1,2,3,4]*AC(man,East,i,j)[4,5,6,7]*fp1LR(man,South,i,j)[7,8,9,10]*AC(man,West,i,j)[10,11,12,1]*
        man.peps[i,j][11,8,5,2,13]*conj(man.peps[i,j][12,9,6,3,14])*opp[14,13]
        n = @tensor fp1LR(man,North,i,j)[1,2,3,4]*AC(man,East,i,j)[4,5,6,7]*fp1LR(man,South,i,j)[7,8,9,10]*AC(man,West,i,j)[10,11,12,1]*
        man.peps[i,j][11,8,5,2,13]*conj(man.peps[i,j][12,9,6,3,13])

        e/n
    end
end

function MPSKit.expectation_value(man::InfEnvManager,nn::NN)
    tot = 0.0+0im

    for i = 1:size(man.peps,1), j = 1:size(man.peps,2)

        tot += @tensor fp1RL(man,North,i,j)[1,2,3,4]*
            AR(man,East,i,j)[4,5,6,7]*
            AR(man,East,i+1,j)[7,8,9,10]*
            fp1LR(man,South,i+1,j)[10,11,12,13]*
            AL(man,West,i+1,j)[13,14,15,16]*
            AL(man,West,i,j)[16,17,18,1]*
            man.peps[i,j][17,19,5,2,20]*
            conj(man.peps[i,j][18,21,6,3,22])*
            man.peps[i+1,j][14,11,8,19,23]*
            conj(man.peps[i+1,j][15,12,9,21,24])*
            nn.o[22,20,24,23]

        tot += @tensor fp1RL(man,West,i,j)[1,2,3,4]*
                AR(man,North,i,j+1)[7,8,9,10]*
                AR(man,North,i,j)[4,5,6,7]*
                fp1LR(man,East,i,j+1)[10,11,12,13]*
                AL(man,South,i,j+1)[13,14,15,16]*
                AL(man,South,i,j)[16,17,18,1]*
                man.peps[i,j][2,17,19,5,20]*
                conj(man.peps[i,j][3,18,21,6,22])*
                man.peps[i,j+1][19,14,11,8,23]*
                conj(man.peps[i,j+1][21,15,12,9,24])*
                nn.o[22,20,24,23]
    end

    tot
end

function MPSKit.expectation_value(man::InfEnvManager,nn::NNN)
    tot = 0.0+0im

    for i = 1:size(man.peps,1), j = 1:size(man.peps,2)
        tot += @tensor fp1LR(man,West,i,j)[21,22,18,17]*
            AL(man,North,i,j)[17,24,19,27]*
            AC(man,North,i,j+1)[27,13,15,11]*
            fp1LR(man,East,i,j+1)[11,12,14,31]*
            corner(man,SouthEast,i,j+1)[31,2]*
            AR(man,East,i+1,j+1)[2,5,10,3]*
            fp1LR(man,South,i+1,j+1)[3,4,9,6]*
            AL(man,West,i+1,j+1)[6,7,8,1]*
            corner(man,SouthWest,i,j+1)[1,29]*
            AR(man,South,i,j)[29,23,20,21]*
            man.peps[i,j][22,23,26,24,33]*
            conj(man.peps[i,j][18,20,25,19,32])*
            man.peps[i,j+1][26,30,12,13,16]*
            conj(man.peps[i,j+1][25,28,14,15,16])*
            man.peps[i+1,j+1][7,4,5,30,35]*
            conj(man.peps[i+1,j+1][8,9,10,28,34])*
            nn.o[32,33,34,35]

        tot += @tensor fp1LR(man,West,i,j)[6,7,3,2]*
            AL(man,North,i,j)[2,9,4,1]*
            corner(man,NorthWest,i,j+1)[1,31]*
            AR(man,West,i-1,j+1)[31,13,15,14]*
            fp1LR(man,North,i-1,j+1)[14,12,17,10]*
            AL(man,East,i-1,j+1)[10,11,16,27]*
            AC(man,East,i,j+1)[27,21,23,19]*
            fp1LR(man,South,i,j+1)[19,20,22,18]*
            corner(man,SouthWest,i,j+1)[18,29]*
            AR(man,South,i,j)[29,8,5,6]*
            man.peps[i,j][7,8,28,9,35]*
            conj(man.peps[i,j][3,5,30,4,34])*
            man.peps[i,j+1][28,20,21,26,24]*
            conj(man.peps[i,j+1][30,22,23,25,24])*
            man.peps[i-1,j+1][13,26,11,12,33]*
            conj(man.peps[i-1,j+1][15,25,16,17,32])*
            nn.o[32,33,34,35]
    end

    tot
end


function MPSKit.expectation_value(man::FinEnvManager,nn::NN)
    #=
    contrast it with the infpeps code. We only had to add bound checks and normalization (ipeps is normalized in place)
    =#

    tot = 0.0+0im
    normalization = 0.0+0im;
    normalcount = 0;
    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))
        if i < size(man.peps,1)
            tot += @tensor fp1RL(man,North,i,j)[1,2,3,4]*
                AR(man,East,i,j)[4,5,6,7]*
                AR(man,East,i+1,j)[7,8,9,10]*
                fp1LR(man,South,i+1,j)[10,11,12,13]*
                AL(man,West,i+1,j)[13,14,15,16]*
                AL(man,West,i,j)[16,17,18,1]*
                man.peps[i,j][17,19,5,2,20]*
                conj(man.peps[i,j][18,21,6,3,22])*
                man.peps[i+1,j][14,11,8,19,23]*
                conj(man.peps[i+1,j][15,12,9,21,24])*
                nn.o[22,20,24,23]

            normalcount +=1;
            normalization += @tensor fp1RL(man,North,i,j)[1,2,3,4]*
            AR(man,East,i,j)[4,5,6,7]*
            AR(man,East,i+1,j)[7,8,9,10]*
            fp1LR(man,South,i+1,j)[10,11,12,13]*
            AL(man,West,i+1,j)[13,14,15,16]*
            AL(man,West,i,j)[16,17,18,1]*
            man.peps[i,j][17,19,5,2,20]*
            conj(man.peps[i,j][18,21,6,3,20])*
            man.peps[i+1,j][14,11,8,19,23]*
            conj(man.peps[i+1,j][15,12,9,21,23])
        end

        if j < size(man.peps,2)
            tot += @tensor fp1RL(man,West,i,j)[1,2,3,4]*
                AR(man,North,i,j+1)[7,8,9,10]*
                AR(man,North,i,j)[4,5,6,7]*
                fp1LR(man,East,i,j+1)[10,11,12,13]*
                AL(man,South,i,j+1)[13,14,15,16]*
                AL(man,South,i,j)[16,17,18,1]*
                man.peps[i,j][2,17,19,5,20]*
                conj(man.peps[i,j][3,18,21,6,22])*
                man.peps[i,j+1][19,14,11,8,23]*
                conj(man.peps[i,j+1][21,15,12,9,24])*
                nn.o[22,20,24,23]

            normalcount +=1;
            normalization += @tensor fp1RL(man,West,i,j)[1,2,3,4]*
            AR(man,North,i,j+1)[7,8,9,10]*
            AR(man,North,i,j)[4,5,6,7]*
            fp1LR(man,East,i,j+1)[10,11,12,13]*
            AL(man,South,i,j+1)[13,14,15,16]*
            AL(man,South,i,j)[16,17,18,1]*
            man.peps[i,j][2,17,19,5,20]*
            conj(man.peps[i,j][3,18,21,6,20])*
            man.peps[i,j+1][19,14,11,8,23]*
            conj(man.peps[i,j+1][21,15,12,9,23])
        end
    end

    normalcount*tot/normalization
end


#=
This is a bit poorly defined
=#
function MPSKit.expectation_value(man::WinEnvManager,nn::NN)

    tot = 0.0+0im
    normalization = 0.0+0im;
    normalcount = 0;
    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))
        if i < size(man.peps,1)
            tot += @tensor fp1RL(man,North,i,j)[1,2,3,4]*
                AR(man,East,i,j)[4,5,6,7]*
                AR(man,East,i+1,j)[7,8,9,10]*
                fp1LR(man,South,i+1,j)[10,11,12,13]*
                AL(man,West,i+1,j)[13,14,15,16]*
                AL(man,West,i,j)[16,17,18,1]*
                man.peps[i,j][17,19,5,2,20]*
                conj(man.peps[i,j][18,21,6,3,22])*
                man.peps[i+1,j][14,11,8,19,23]*
                conj(man.peps[i+1,j][15,12,9,21,24])*
                nn.o[22,20,24,23]

            normalcount +=1;
            normalization += @tensor fp1RL(man,North,i,j)[1,2,3,4]*
            AR(man,East,i,j)[4,5,6,7]*
            AR(man,East,i+1,j)[7,8,9,10]*
            fp1LR(man,South,i+1,j)[10,11,12,13]*
            AL(man,West,i+1,j)[13,14,15,16]*
            AL(man,West,i,j)[16,17,18,1]*
            man.peps[i,j][17,19,5,2,20]*
            conj(man.peps[i,j][18,21,6,3,20])*
            man.peps[i+1,j][14,11,8,19,23]*
            conj(man.peps[i+1,j][15,12,9,21,23])
        end

        if j < size(man.peps,2)
            tot += @tensor fp1RL(man,West,i,j)[1,2,3,4]*
                AR(man,North,i,j+1)[7,8,9,10]*
                AR(man,North,i,j)[4,5,6,7]*
                fp1LR(man,East,i,j+1)[10,11,12,13]*
                AL(man,South,i,j+1)[13,14,15,16]*
                AL(man,South,i,j)[16,17,18,1]*
                man.peps[i,j][2,17,19,5,20]*
                conj(man.peps[i,j][3,18,21,6,22])*
                man.peps[i,j+1][19,14,11,8,23]*
                conj(man.peps[i,j+1][21,15,12,9,24])*
                nn.o[22,20,24,23]

            normalcount +=1;
            normalization += @tensor fp1RL(man,West,i,j)[1,2,3,4]*
            AR(man,North,i,j+1)[7,8,9,10]*
            AR(man,North,i,j)[4,5,6,7]*
            fp1LR(man,East,i,j+1)[10,11,12,13]*
            AL(man,South,i,j+1)[13,14,15,16]*
            AL(man,South,i,j)[16,17,18,1]*
            man.peps[i,j][2,17,19,5,20]*
            conj(man.peps[i,j][3,18,21,6,20])*
            man.peps[i,j+1][19,14,11,8,23]*
            conj(man.peps[i,j+1][21,15,12,9,23])
        end
    end

    normalcount*tot/normalization
end
