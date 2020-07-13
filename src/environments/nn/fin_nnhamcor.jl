mutable struct FinNNHamCors{E<:FinEnvManager,B,C,O<:NN} <: Cache
    opperator :: O
    envm::E

    cors::B # in principle we also need partially contracted cors ...
    lines::C
end

function correlator(envm::FinEnvManager,o::NN)
    lines = similar(envm.fp1);
    cors = similar(envm.boundaries);

    for dir in Dirs
        lines[dir] = zero.(envm.fp1[dir])
        cors[dir] = copy.(envm.boundaries[dir])
        cors[dir][1] *= 0;
    end

    tor = FinNNHamCors(o,envm,cors,lines);

    MPSKit.recalculate!(tor,envm.peps);
end

function MPSKit.recalculate!(corenvs::FinNNHamCors,peps::FinPEPS)
    MPSKit.recalculate!(corenvs.envm,peps);

    recalc_lines!(corenvs);
    recalc_cors!(corenvs);

    corenvs
end

function recalc_cors!(corenvs::FinNNHamCors)
    @sync for dir in Dirs
        @Threads.spawn begin
            tpeps = rotate_north(corenvs.envm.peps,dir);
            for i in 1:size(tpeps,1)
                (corenvs.cors[dir][i+1],_) = approximate(corenvs.cors[dir][i+1],
                                            [(tpeps[i,:],corenvs.opperator,corenvs.envm.boundaries[dir][i]),
                                            (tpeps[i,:],corenvs.cors[dir][i])],
                                            corenvs.envm.algorithm);
            end
        end
    end
end

function recalc_lines!(env::FinNNHamCors)
    for dir in Dirs
        tman = rotate_north(env.envm,dir);
        tpeps = tman.peps;

        for i = 2:size(tpeps,1)
            for j = 1:size(tpeps,2)
                env.lines[dir][i+1,j] = crosstransfer(env.lines[dir][i,j],tpeps[i,j],AL(tman,East,i,j),AR(tman,West,i,j));
                env.lines[dir][i+1,j] += hamtransfer(AR(tman,West,i,j),AR(tman,West,i-1,j),AL(tman,East,i-1,j),AL(tman,East,i,j),fp1LR(tman,North,i-1,j),tpeps[i-1,j],tpeps[i,j],env.opperator)
            end
        end
    end
end
