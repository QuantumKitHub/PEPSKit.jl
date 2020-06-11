mutable struct FinNNHamChannels{E<:FinEnvManager,B,O<:NN} <: Cache
    opperator :: O
    envm::E

    lines::B
    ts::B
end

#generate bogus data
function MPSKit.params(peps::FinPEPS,opperator::NN,alg::MPSKit.Algorithm)
    pepspars = params(peps,alg);

    lines = similar(pepspars.fp1);
    ts = similar(pepspars.fp1);

    for dir in Dirs
        lines[dir] = zero.(pepspars.fp1[dir])
        ts[dir] = zero.(pepspars.fp1[dir])
    end

    pars = FinNNHamChannels(opperator,pepspars,lines,ts);

    return MPSKit.recalculate!(pars,peps)
end

#recalculate everything
function MPSKit.recalculate!(prevenv::FinNNHamChannels,peps::FinPEPS)
    MPSKit.recalculate!(prevenv.envm,peps);

    recalc_lines!(prevenv)
    recalc_ts!(prevenv)
    prevenv
end

function recalc_lines!(env::FinNNHamChannels)
    for dir in Dirs
        tman = rotate_north(env.envm,dir);
        tpeps = tman.peps;

        for i = 2:size(tpeps,1)
            for j = 1:size(tpeps,2)
                #notice just how similar this is to the infinite peps case
                #I don't subtract any fps yet, maybe later?
                env.lines[dir][i+1,j] = crosstransfer(env.lines[dir][i,j],tpeps[i,j],AL(tman,East,i,j),AR(tman,West,i,j));
                env.lines[dir][i+1,j] += hamtransfer(AR(tman,West,i,j),AR(tman,West,i-1,j),AL(tman,East,i-1,j),AL(tman,East,i,j),fp1LR(tman,North,i-1,j),tpeps[i-1,j],tpeps[i,j],env.opperator)
            end
        end
    end
end

function recalc_ts!(env::FinNNHamChannels)
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
                if i > 1
                    @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                        fp1RL(man,North,i-1,j)[1,2,3,4]*
                        AL(man,West,i-1,j)[5,6,7,1]*
                        AR(man,East,i-1,j)[4,10,11,12]*
                        AL(man,West,i,j)[-1,8,9,5]*
                        AR(man,East,i,j)[12,13,14,-4]*
                        man.peps[i-1,j][6,15,10,2,16]*
                        conj(man.peps[i-1,j][7,17,11,3,18])*
                        man.peps[i,j][8,-2,13,15,19]*
                        conj(man.peps[i,j][9,-3,14,17,20])*
                        nn[16,18,19,20]
                end

                # "horleft contribution"
                if j > 1
                    @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                        fp1LR(man,North,i,j)[1,4,6,2]*
                        AC(man,East,i,j)[2,3,5,-4]*
                        corner(man,NorthWest,i,j)[22,1]*
                        AL(man,North,i,j-1)[8,10,12,22]*
                        fp1LR(man,West,i,j-1)[15,9,11,8]*
                        AR(man,South,i,j-1)[7,13,14,15]*
                        corner(man,SouthWest,i,j)[-1,7]*
                        man.peps[i,j-1][9,13,20,10,16]*
                        conj(man.peps[i,j-1][11,14,18,12,17])*
                        man.peps[i,j][20,-2,3,4,21]*
                        conj(man.peps[i,j][18,-3,5,6,19])*
                        nn[16,17,21,19]
                end

                # "horright contribution"
                if j < size(tpeps,2)
                    @tensor env.ts[dir][i+1,j][-1 -2 -3;-4]+=
                        AC(man,West,i,j)[-1,3,5,2]*
                        fp1LR(man,North,i,j)[2,4,6,1]*
                        corner(man,NorthEast,i,j)[1,22]*
                        fp1LR(man,East,i,j+1)[8,9,11,13]*
                        AR(man,North,i,j+1)[22,10,12,8]*
                        AL(man,South,i,j+1)[13,14,15,7]*
                        corner(man,SouthEast,i,j)[7,-4]*
                        man.peps[i,j][3,-2,20,4,21]*
                        conj(man.peps[i,j][5,-3,18,6,19])*
                        man.peps[i,j+1][20,14,9,10,16]*
                        conj(man.peps[i,j+1][18,15,11,12,17])*
                                        nn[21,19,16,17]
                end
            end
        end

    end
end
