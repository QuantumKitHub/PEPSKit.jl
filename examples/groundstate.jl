using MPSKit,PEPSKit,TensorKit,OptimKit

#=
    This function will be called in the middle of the boundary-mps-optimization
    Here we can mess with the parameters a bit (like increasing the bond dimension)
=#
function bound_finalize(iter,state,ham,pars)
    #=
        We are in the middle of the boundary-mps-optimization
        Here we can mess with the parameters a bit (like increasing the bond dimension)
    =#

    maxD = 50;

    # check every virtual bond dimension and see if it's larger then maxD
    bigenough = true;
    for i in 1:size(state,1)
        for j in 1:size(state,2)
            bigenough = bigenough && (dim(MPSKit.virtualspace(state,i,j)) >= maxD)
        end
    end


    #it's not larger then maxD ? change the bond dimension!
    if !bigenough
        (state,pars) = changebonds(state,ham,OptimalExpand(trscheme=truncdim(10)),pars)
    end

    return (state,pars)
end

ham = nonsym_nn_xxz_ham();

# 2 by 2 unit cell
peps = InfPEPS(map(Iterators.product(1:2,1:2)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',ℂ^2)
end);

optalg = LBFGS(linesearch = HagerZhangLineSearch(ϵ = 1e-3,verbosity=0),maxiter=50,verbosity=20,gradtol=1e-3)
(peps,pars,err) = find_groundstate(peps,ham,optalg,bound_finalize=bound_finalize)

@show expectation_value(peps,ham,pars)
