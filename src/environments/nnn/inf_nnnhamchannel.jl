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

    @tensor v[-1 -2 -3;-4] += fp1LR(envm,North,row-1,col)[1,2,3,4]*
        corner(envm,NorthEast,row-1,col)[4,5]*
        AR(envm,North,row-1,col+1)[5,6,7,8]*
        fp1LR(envm,East,row-1,col+1)[8,9,10,11]*
        AL(envm,South,row-1,col+1)[11,12,13,14]*
        corner(envm,SouthEast,row-1,col)[14,15]*
        AR(envm,East,row,col)[15,16,17,-4]*
        AR(envm,West,row,col)[-1,18,19,20]*
        AR(envm,West,row-1,col)[20,22,23,1]*
        peps[row,col][18,-2,16,24,26]*
        conj(peps[row,col][19,-3,17,25,27])*
        peps[row-1,col][22,24,28,2,30]*
        conj(peps[row-1,col][23,25,29,3,30])*
        peps[row-1,col+1][28,12,9,6,31]*
        conj(peps[row-1,col+1][29,13,10,7,32])*
        opp.o[32,31,27,26]
end

function north_RU_local(v,envm,opp::NNN,row,col)
    peps = envm.peps;
    v = crosstransfer(v,peps[row,col],AL(envm,East,row,col),AL(envm,West,row,col));

    @tensor v[-1 -2 -3;-4] += AL(envm,West,row,col)[-1,1,2,3]*
        corner(envm,SouthWest,row-1,col)[3,4]*
        AR(envm,South,row-1,col-1)[4,5,6,7]*
        fp1LR(envm,West,row-1,col-1)[7,8,9,10]*
        AL(envm,North,row-1,col-1)[10,11,12,13]*
        corner(envm,NorthWest,row-1,col)[13,14]*
        fp1LR(envm,North,row-1,col)[14,15,16,17]*
        AL(envm,East,row-1,col)[17,18,19,20]*
        AL(envm,East,row,col)[20,21,22,-4]*
        peps[row,col][1,-2,21,23,25]*
        conj(peps[row,col][2,-3,22,24,26])*
        peps[row-1,col][27,23,18,15,29]*
        conj(peps[row-1,col][28,24,19,16,29])*
        peps[row-1,col-1][8,5,27,11,30]*
        conj(peps[row-1,col-1][9,6,28,12,31])*
        opp.o[31,30,26,25]
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
        LU[1,2,3,4]*
        AR(envm,North,row,col)[4,5,6,7]*
        fp1LR(envm,East,row,col)[7,8,9,10]*
        corner(envm,SouthEast,row,col)[10,-4]*
        peps[row,col][2,-2,8,5,11]*
        conj(peps[row,col][3,-3,9,6,11])

    #add RU contribution (LL gauged)
    @tensor v[-1 -2 -3;-4] += corner(envm,SouthWest,row,col)[-1,1]*
        fp1LR(envm,West,row,col)[1,2,3,4]*
        AL(envm,North,row,col)[4,5,6,7]*
        RU[7,8,9,10]*
        corner(envm,SouthEast,row,col)[10,-4]*
        peps[row,col][2,-2,8,5,11]*
        conj(peps[row,col][3,-3,9,6,11])

    #add /.
    @tensor v[-1 -2 -3;-4] += corner(envm,SouthWest,row,col)[-1,1]*
        AR(envm,South,row,col-1)[1,2,3,4]*
        fp1LR(envm,West,row,col-1)[4,5,6,7]*
        AL(envm,North,row,col-1)[7,8,9,10]*
        corner(envm,NorthWest,row,col)[10,11]*
        AR(envm,West,row-1,col)[11,12,13,14]*
        fp1LR(envm,North,row-1,col)[14,15,16,17]*
        AL(envm,East,row-1,col)[17,18,19,20]*
        AC(envm,East,row,col)[20,21,22,-4]*
        peps[row,col][23,-2,21,25,27]*
        conj(peps[row,col][24,-3,22,26,27])*
        peps[row,col-1][5,2,23,8,28]*
        conj(peps[row,col-1][6,3,24,9,29])*
        peps[row-1,col][12,25,18,15,30]*
        conj(peps[row-1,col][13,26,19,16,31])*
        opp.o[29,28,31,30]

    #add .\
    @tensor v[-1 -2 -3;-4] += AC(envm,West,row,col)[-1,1,2,3]*
        AR(envm,West,row-1,col)[3,4,5,6]*
        fp1LR(envm,North,row-1,col)[6,7,8,9]*
        AL(envm,East,row-1,col)[9,10,11,12]*
        corner(envm,NorthEast,row,col)[12,13]*
        AR(envm,North,row,col+1)[13,14,15,16]*
        fp1LR(envm,East,row,col+1)[16,17,18,19]*
        AL(envm,South,row,col+1)[19,20,21,22]*
        corner(envm,SouthEast,row,col)[22,-4]*
        peps[row,col][1,-2,23,25,27]*
        conj(peps[row,col][2,-3,24,26,27])*
        peps[row-1,col][4,25,10,7,28]*
        conj(peps[row-1,col][5,26,11,8,29])*
        peps[row,col+1][23,20,17,14,30]*
        conj(peps[row,col+1][24,21,18,15,31])*
        opp.o[29,28,31,30]

    #add \ (copy-pasted, so should be designed differently)
    @tensor v[-1 -2 -3;-4] += fp1LR(envm,North,row-1,col)[1,2,3,4]*
        corner(envm,NorthEast,row-1,col)[4,5]*
        AR(envm,North,row-1,col+1)[5,6,7,8]*
        fp1LR(envm,East,row-1,col+1)[8,9,10,11]*
        AL(envm,South,row-1,col+1)[11,12,13,14]*
        corner(envm,SouthEast,row-1,col)[14,15]*
        AR(envm,East,row,col)[15,16,17,-4]*
        AC(envm,West,row,col)[-1,18,19,20]*
        AR(envm,West,row-1,col)[20,22,23,1]*
        peps[row,col][18,-2,16,24,26]*
        conj(peps[row,col][19,-3,17,25,27])*
        peps[row-1,col][22,24,28,2,30]*
        conj(peps[row-1,col][23,25,29,3,30])*
        peps[row-1,col+1][28,12,9,6,31]*
        conj(peps[row-1,col+1][29,13,10,7,32])*
        opp.o[32,31,27,26]

    #add / (also copy pasted)
    @tensor v[-1 -2 -3;-4] += AL(envm,West,row,col)[-1,1,2,3]*
        corner(envm,SouthWest,row-1,col)[3,4]*
        AR(envm,South,row-1,col-1)[4,5,6,7]*
        fp1LR(envm,West,row-1,col-1)[7,8,9,10]*
        AL(envm,North,row-1,col-1)[10,11,12,13]*
        corner(envm,NorthWest,row-1,col)[13,14]*
        fp1LR(envm,North,row-1,col)[14,15,16,17]*
        AL(envm,East,row-1,col)[17,18,19,20]*
        AC(envm,East,row,col)[20,21,22,-4]*
        peps[row,col][1,-2,21,23,25]*
        conj(peps[row,col][2,-3,22,24,26])*
        peps[row-1,col][27,23,18,15,29]*
        conj(peps[row-1,col][28,24,19,16,29])*
        peps[row-1,col-1][8,5,27,11,30]*
        conj(peps[row-1,col-1][9,6,28,12,31])*
        opp.o[31,30,26,25]
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
