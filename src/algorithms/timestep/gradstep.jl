struct GradStep <: MPSKit.Algorithm
end

function timestep(state::FinPEPS, H::NN, timestep::Number,alg::GradStep,pars::FinNNHamChannels)
    newpeps = copy(state);
    for (i,j) in Iterators.product(1:size(state,1),1:size(state,2))
        (h_eff,n_eff) = effectivehn(pars,i,j)
        v = permute(state[i,j],(1,2,3,4,5))

        n = dot(v,n_eff*v);
        h = dot(v,h_eff*v);

        g = (h_eff*v-(h/n)*n_eff*v)/n
        newpeps[i,j]+=-1im*timestep*permute(g,(1,2,3,4),(5,));
    end

    newpars = deepcopy(pars);

    MPSKit.recalculate!(newpars,newpeps)

    return newpeps,newpars
end
