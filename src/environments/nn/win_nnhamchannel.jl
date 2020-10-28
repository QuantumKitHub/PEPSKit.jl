mutable struct WinNNHamChannels{E<:WinEnvManager,B,O<:NN,C} <: Cache
    opperator :: O
    envm::E

    infchan::C

    lines::B
    ts::B
end

#generate bogus data
function channels(envm::WinEnvManager,operator::NN)
    peps = envm.peps;

    lines = similar.(envm.fp1);
    ts = similar.(envm.fp1);

    infchan = channels(envm.infenvm,operator);

    pars = WinNNHamChannels(operator,envm,infchan,lines,ts);

    return recalculate!(pars,pars.envm)
end

function MPSKit.recalculate!(chan::WinNNHamChannels,envm::WinEnvManager)
    chan.envm = envm;
    recalc_lines!(chan)
    recalc_ts!(chan)
    chan
end
#recalculate everything
function MPSKit.recalculate!(chan::WinNNHamChannels,peps::WinPEPS)
    recalculate!(chan.envm,peps);
    recalculate!(chan,chan.envm);
end

function recalc_lines!(env::WinNNHamChannels)
    for dir in Dirs
        tman = rotate_north(env.envm,dir);
        tpeps = tman.peps;

        for j = 1:size(tpeps,2)
            env.lines[dir][1,j] = env.infchan.lines[dir][1,j]
            for i = 1:size(tpeps,1)
                #notice just how similar this is to the infinite peps case
                #I don't subtract any fps yet, maybe later?
                env.lines[dir][i+1,j] = crosstransfer(env.lines[dir][i,j],tpeps[i,j],AL(tman,East,i,j),AR(tman,West,i,j));
                env.lines[dir][i+1,j] += hamtransfer(AR(tman,West,i,j),AR(tman,West,i-1,j),AL(tman,East,i-1,j),AL(tman,East,i,j),fp1LR(tman,North,i-1,j),tpeps[i-1,j],tpeps[i,j],env.opperator)
            end
        end
    end
end

function recalc_ts!(env::WinNNHamChannels)
    #fix the initial ts
    for dir in Dirs
        peps = env.envm.peps;
        (tnr,tnc) = rotate_north(size(peps),dir)
        envm = env.envm;
        for j = 1:tnc
            env.ts[dir][1,j] = env.infchan.ts[dir][1,j]

            #correct C
            (ti,tj) = inv_rotate_north((1,j),size(peps),dir);

            C1_e = envm.infenvm.boundaries[left(dir)].CR[j,end-1+1]
            C1_i = envm.boundaries[left(dir)][j].CR[end-1+1]
            C2_e = envm.infenvm.boundaries[right(dir)].CR[end-j+1,1-1]
            C2_i = envm.boundaries[right(dir)][end-j].CR[1-1]

            @tensor env.ts[dir][1,j][-1 -2 -3;-4] := env.ts[dir][1,j][1 -2 -3;2]*inv(C1_e)[3,1]*C1_i[-1,3]*inv(C2_e)[2,4]*C2_i[4,-4]
        end

    end

    #lines are already updated here :)
    for dir in Dirs
        man = rotate_north(env.envm,dir);
        tpeps = man.peps;
        nn = env.opperator
        for i in 1:size(tpeps,1)

            for j = 1:size(tpeps,2)
                env.ts[dir][i+1,j] = crosstransfer(env.ts[dir][i,j],tpeps[i,j],AR(man,East,i,j),AL(man,West,i,j));

                #collect west and east contributions from lines
                (wi,wj) = rotate_north((i,j),size(tpeps),West);
                (ei,ej) = rotate_north((i,j),size(tpeps),East);

                cwcontr = env.lines[left(dir)][wi,wj];
                cecontr = env.lines[right(dir)][ei,ej];

                # "add west contribution"
                @tensor env.ts[dir][i+1,j][-1 -2 -3;-4] +=
                    corner(man,SouthWest,i,j)[-1,2]*
                    cwcontr[2,10,11,1]*
                    corner(man,NorthWest,i,j)[1,9]*
                    fp1LR(man,North,i,j)[9,5,7,3]*
                    AC(man,East,i,j)[3,4,6,-4]*
                    tpeps[i,j][10,-2,4,5,8]*
                    conj(tpeps[i,j][11,-3,6,7,8])

                # "add east contribution"
                @tensor env.ts[dir][i+1,j][-1 -2 -3;-4] +=
                    AC(man,West,i,j)[-1,4,6,3]*
                    fp1LR(man,North,i,j)[3,5,7,9]*
                    corner(man,NorthEast,i,j)[9,1]*
                    cecontr[1,10,11,2]*
                    corner(man,SouthEast,i,j)[2,-4]*
                    man.peps[i,j][4,-2,10,5,8]*
                    conj(man.peps[i,j][6,-3,11,7,8])

                # "vertical ham contribution"
                @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                    fp1LR(man,North,i-1,j)[8,3,5,1]*
                    AR(man,West,i-1,j)[20,6,7,8]*
                    AL(man,East,i-1,j)[1,2,4,11]*
                    AC(man,West,i,j)[-1,18,19,20]*
                    AC(man,East,i,j)[11,12,15,-4]*
                    man.peps[i-1,j][6,13,2,3,9]*
                    conj(man.peps[i-1,j][7,16,4,5,10])*
                    man.peps[i,j][18,-2,12,13,14]*
                    conj(man.peps[i,j][19,-3,15,16,17])*
                    nn[10,9,17,14]

                # "horleft contribution"
                @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                    fp1LR(man,North,i,j)[14,15,17,19]*
                    AC(man,East,i,j)[19,20,21,-4]*
                    corner(man,NorthWest,i,j)[1,14]*
                    AL(man,North,i,j-1)[2,4,6,1]*
                    fp1LR(man,West,i,j-1)[9,3,5,2]*
                    AR(man,South,i,j-1)[22,7,8,9]*
                    corner(man,SouthWest,i,j)[-1,22]*
                    man.peps[i,j-1][3,7,12,4,10]*
                    conj(man.peps[i,j-1][5,8,16,6,11])*
                    man.peps[i,j][12,-2,20,15,13]*
                    conj(man.peps[i,j][16,-3,21,17,18])*
                    nn[11,10,18,13]

                # "horright contribution"
                @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                    AC(man,West,i,j)[-1,20,21,22]*
                    fp1LR(man,North,i,j)[22,15,18,16]*
                    corner(man,NorthEast,i,j)[16,2]*
                    fp1LR(man,East,i,j+1)[3,4,6,8]*
                    AR(man,North,i,j+1)[2,5,7,3]*
                    AL(man,South,i,j+1)[8,9,10,1]*
                    corner(man,SouthEast,i,j)[1,-4]*
                    man.peps[i,j][20,-2,13,15,14]*
                    conj(man.peps[i,j][21,-3,17,18,19])*
                    man.peps[i,j+1][13,9,4,5,11]*
                    conj(man.peps[i,j+1][17,10,6,7,12])*
                    nn[19,14,12,11]

            end
        end

    end
end
