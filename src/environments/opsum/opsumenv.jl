mutable struct OpSumEnv{O<:OpSum,B<:Tuple,C} <: Cache
    opperator :: O
    envm :: C
    envs :: B
end

function channels(envm,opperator::OpSum)
    envs = map(o->channels(envm,o),opperator.ops)
    OpSumEnv(opperator,envm,envs)
end

function MPSKit.recalculate!(chan::OpSumEnv,env::InfEnvManager)
    chan.envm = env;

    @sync for E in chan.envs
        @Threads.spawn recalculate!(E,env)
    end
    
    chan
end

function MPSKit.recalculate!(chan::OpSumEnv,peps::InfPEPS)
    recalculate!(chan.envm,peps);
    recalculate!(chan,chan.envm);
end
