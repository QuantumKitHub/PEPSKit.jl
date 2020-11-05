mutable struct InfNNNHamChannels{E<:InfEnvManager,B,O<:NNN} <: Cache
    opperator :: O
    envm::E

    LU::B
    RU::B
    ts::B
end

#generate bogus data
function channels(envm::InfEnvManager,opperator::NNN)
    peps = envm.peps

    LU = PeriodicArray(similar.(envm.fp1));
    RU = PeriodicArray(similar.(envm.fp1));
    ts = PeriodicArray(similar.(envm.fp1));

    chan = InfNNNHamChannels(opperator,envm,LU,RU,ts);
    recalculate!(chan,envm);
end

function MPSKit.recalculate!(chan::InfNNNHamChannels,env::InfEnvManager)
    chan.envm = env;

    @sync for dir in Dirs
        tenv = rotate_north(env,dir);

        @Threads.spawn north_LU!(chan.LU[dir],tenv,chan.opperator)
        @Threads.spawn north_RU!(chan.RU[dir],tenv,chan.opperator)
    end

    @sync for dir in Dirs
        tenv = rotate_north(env,dir);

        @Threads.spawn north_nnntchannel!(chan.ts[dir],tenv,rotl90(chan.LU[left(dir)]),rotr90(chan.RU[right(dir)]),chan.opperator);
    end

    chan
end

function MPSKit.recalculate!(chan::InfNNNHamChannels,peps::InfPEPS)
    recalculate!(chan.envm,peps);
    recalculate!(chan,chan.envm);
end

function north_inf_sum_RR(v,man,j)
    downfp = fp1RR(man,South,0,j);
    upfp = fp1RR(man,North,1,j);
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y = crosstransfer(x,man.peps[:,j],AR(man,East,:,j),AR(man,West,:,j))
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*(upfp)[-1,-2,-3,-4]

        x-y
    end
    chist.converged == 0 && @warn "failed to converge _ LR $(chist.normres)"

    return v
end

function north_inf_sum_LL(v,man,j)
    downfp = fp1LL(man,South,0,j);
    upfp = fp1LL(man,North,1,j);
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y = crosstransfer(x,man.peps[:,j],AL(man,East,:,j),AL(man,West,:,j))
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*(upfp)[-1,-2,-3,-4]

        x-y
    end
    chist.converged == 0 && @warn "failed to converge _ LR $(chist.normres)"

    return v
end

function north_LU_local(v,envm,opp::NNN,row,col)
    peps = envm.peps;
    v = crosstransfer(v,peps[row,col],AR(envm,East,row,col),AR(envm,West,row,col));

    @tensor v[-1 -2 -3;-4] += fp1LR(envm,North,row-1,col)[2,6,4,1]*
        corner(envm,NorthEast,row-1,col)[1,21]*
        AR(envm,North,row-1,col+1)[21,16,12,13]*
        fp1LR(envm,East,row-1,col+1)[13,15,11,9]*
        AL(envm,South,row-1,col+1)[9,14,10,8]*
        corner(envm,SouthEast,row-1,col)[8,24]*
        AR(envm,East,row,col)[24,25,26,-4]*
        AR(envm,West,row,col)[-1,29,30,31]*
        AR(envm,West,row-1,col)[31,5,3,2]*
        peps[row,col][29,-2,25,22,23]*
        conj(peps[row,col][30,-3,26,27,28])*
        peps[row-1,col][5,22,19,6,7]*
        conj(peps[row-1,col][3,27,20,4,7])*
        peps[row-1,col+1][19,14,15,16,18]*
        conj(peps[row-1,col+1][20,10,11,12,17])*
        opp.o[17,18,28,23]
end

function north_RU_local(v,envm,opp::NNN,row,col)
    peps = envm.peps;
    v = crosstransfer(v,peps[row,col],AL(envm,East,row,col),AL(envm,West,row,col));

    @tensor v[-1 -2 -3;-4] += AL(envm,West,row,col)[-1,30,31,1]*
        corner(envm,SouthWest,row-1,col)[1,29]*
        AR(envm,South,row-1,col-1)[29,15,11,9]*
        fp1LR(envm,West,row-1,col-1)[9,14,10,12]*
        AL(envm,North,row-1,col-1)[12,16,13,21]*
        corner(envm,NorthWest,row-1,col)[21,2]*
        fp1LR(envm,North,row-1,col)[2,7,5,3]*
        AL(envm,East,row-1,col)[3,6,4,24]*
        AL(envm,East,row,col)[24,25,26,-4]*
        peps[row,col][30,-2,25,22,23]*
        conj(peps[row,col][31,-3,26,27,28])*
        peps[row-1,col][19,22,6,7,8]*
        conj(peps[row-1,col][20,27,4,5,8])*
        peps[row-1,col-1][14,15,19,16,18]*
        conj(peps[row-1,col-1][10,11,20,13,17])*
        opp.o[17,18,28,23]
end

function north_LU!(dst,envm::InfEnvManager,opp::NNN)
    peps = envm.peps;
    for col in 1:size(peps,2)
        #determine local part
        start = fp1LR(envm,North,1,col)*0;
        for row in 1:size(peps,1);
            start = north_LU_local(start,envm,opp,row,col);
        end

        #do transfer inversion
        totransfer = north_inf_sum_RR(start,envm,col);

        #store
        for row in 1:size(peps,1)
            if row == 1
                dst[row,col] = totransfer;
            else
                dst[row,col] = north_LU_local(dst[row-1,col],envm,opp,row-1,col);
            end
        end
    end
end

function north_RU!(dst,envm::InfEnvManager,opp::NNN)
    peps = envm.peps;
    for col in 1:size(peps,2)
        #determine local part
        start = fp1LR(envm,North,1,col)*0;
        for row in 1:size(peps,1);
            start = north_RU_local(start,envm,opp,row,col);
        end

        #do transfer inversion
        totransfer = north_inf_sum_LL(start,envm,col);

        #store
        for row in 1:size(peps,1)
            if row == 1
                dst[row,col] = totransfer;
            else
                dst[row,col] = north_RU_local(dst[row-1,col],envm,opp,row-1,col);
            end
        end
    end
end

function north_nnntchannel_local(v,envm,LU,RU,opp,row,col)
    peps = envm.peps;
    v = crosstransfer(v,peps[row,col],AR(envm,East,row,col),AL(envm,West,row,col));

    #add LU contribution (RR gauged)
    @tensor v[-1 -2 -3;-4] += corner(envm,SouthWest,row,col)[-1,1]*
        LU[1,3,5,2]*
        AR(envm,North,row,col)[2,4,6,8]*
        fp1LR(envm,East,row,col)[8,9,10,11]*
        corner(envm,SouthEast,row,col)[11,-4]*
        peps[row,col][3,-2,9,4,7]*
        conj(peps[row,col][5,-3,10,6,7])

    #add RU contribution (LL gauged)
    @tensor v[-1 -2 -3;-4] += corner(envm,SouthWest,row,col)[-1,1]*
        fp1LR(envm,West,row,col)[1,3,5,2]*
        AL(envm,North,row,col)[2,4,6,8]*
        RU[8,9,10,11]*
        corner(envm,SouthEast,row,col)[11,-4]*
        peps[row,col][3,-2,9,4,7]*
        conj(peps[row,col][5,-3,10,6,7])

    #add /.
    @tensor v[-1 -2 -3;-4] += corner(envm,SouthWest,row,col)[-1,10]*
        AR(envm,South,row,col-1)[10,17,13,11]*
        fp1LR(envm,West,row,col-1)[11,16,12,14]*
        AL(envm,North,row,col-1)[14,18,15,9]*
        corner(envm,NorthWest,row,col)[9,22]*
        AR(envm,West,row-1,col)[22,6,4,5]*
        fp1LR(envm,North,row-1,col)[5,8,3,1]*
        AL(envm,East,row-1,col)[1,7,2,29]*
        AC(envm,East,row,col)[29,30,31,-4]*
        peps[row,col][24,-2,30,25,28]*
        conj(peps[row,col][26,-3,31,27,28])*
        peps[row,col-1][16,17,24,18,20]*
        conj(peps[row,col-1][12,13,26,15,19])*
        peps[row-1,col][6,25,7,8,21]*
        conj(peps[row-1,col][4,27,2,3,23])*
        opp.o[19,20,23,21]

    #add .\
    @tensor v[-1 -2 -3;-4] += AC(envm,West,row,col)[-1,28,29,30]*
        AR(envm,West,row-1,col)[30,15,13,14]*
        fp1LR(envm,North,row-1,col)[14,17,12,10]*
        AL(envm,East,row-1,col)[10,16,11,9]*
        corner(envm,NorthEast,row,col)[9,21]*
        AR(envm,North,row,col+1)[21,8,4,5]*
        fp1LR(envm,East,row,col+1)[5,7,3,1]*
        AL(envm,South,row,col+1)[1,6,2,31]*
        corner(envm,SouthEast,row,col)[31,-4]*
        peps[row,col][28,-2,23,24,27]*
        conj(peps[row,col][29,-3,25,26,27])*
        peps[row-1,col][15,24,16,17,19]*
        conj(peps[row-1,col][13,26,11,12,18])*
        peps[row,col+1][23,6,7,8,20]*
        conj(peps[row,col+1][25,2,3,4,22])*
        opp.o[18,19,22,20]

    #add \ (copy-pasted, so should be designed differently)
    @tensor v[-1 -2 -3;-4] += fp1LR(envm,North,row-1,col)[2,6,4,1]*
        corner(envm,NorthEast,row-1,col)[1,21]*
        AR(envm,North,row-1,col+1)[21,16,12,13]*
        fp1LR(envm,East,row-1,col+1)[13,15,11,9]*
        AL(envm,South,row-1,col+1)[9,14,10,8]*
        corner(envm,SouthEast,row-1,col)[8,24]*
        AR(envm,East,row,col)[24,25,26,-4]*
        AR(envm,West,row,col)[-1,29,30,31]*
        AR(envm,West,row-1,col)[31,5,3,2]*
        peps[row,col][29,-2,25,22,23]*
        conj(peps[row,col][30,-3,26,27,28])*
        peps[row-1,col][5,22,19,6,7]*
        conj(peps[row-1,col][3,27,20,4,7])*
        peps[row-1,col+1][19,14,15,16,18]*
        conj(peps[row-1,col+1][20,10,11,12,17])*
        opp.o[17,18,28,23]


    #add / (also copy pasted)
    @tensor v[-1 -2 -3;-4] += AL(envm,West,row,col)[-1,30,31,1]*
        corner(envm,SouthWest,row-1,col)[1,29]*
        AR(envm,South,row-1,col-1)[29,15,11,9]*
        fp1LR(envm,West,row-1,col-1)[9,14,10,12]*
        AL(envm,North,row-1,col-1)[12,16,13,21]*
        corner(envm,NorthWest,row-1,col)[21,2]*
        fp1LR(envm,North,row-1,col)[2,7,5,3]*
        AL(envm,East,row-1,col)[3,6,4,24]*
        AL(envm,East,row,col)[24,25,26,-4]*
        peps[row,col][30,-2,25,22,23]*
        conj(peps[row,col][31,-3,26,27,28])*
        peps[row-1,col][19,22,6,7,8]*
        conj(peps[row-1,col][20,27,4,5,8])*
        peps[row-1,col-1][14,15,19,16,18]*
        conj(peps[row-1,col-1][10,11,20,13,17])*
        opp.o[17,18,28,23]

end

function north_nnntchannel!(dst,envm::InfEnvManager,LU,RU,opp::NNN)
    peps = envm.peps;

    for col in 1:size(peps,2)
        #determine local part
        start = fp1LR(envm,North,1,col)*0;
        for row in 1:size(peps,1)
            start = north_nnntchannel_local(start,envm,LU[row,col],RU[row,col],opp,row,col)
        end

        #do transfer inversion
        ect = north_inf_sum_RL(start,envm,col)

        #store
        for row in 1:size(peps,1)
            if row == 1
                dst[row,col] = ect;
            else
                dst[row,col] = north_nnntchannel_local(dst[row-1,col],envm,LU[row-1,col],RU[row-1,col],opp,row-1,col);
            end
        end
    end
end
