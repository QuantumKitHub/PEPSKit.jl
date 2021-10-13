# This notebook contains a brief overview on how pepskit handles infinite peps.
using Revise,MPSKit,PEPSKit,TensorKit,OptimKit,Plots

#%%

#=
 The infinite peps constructor takes a 2 dimensional array of peps tensors.
 The following creates a 2 by 2 unit cell peps.
=#

data = map(Iterators.product(1:2,1:2)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',(ℂ^2)')
end
peps = InfPEPS(data);

#%%

# To actually calculate anything meaningful, we need access to the peps environments.
# This object can be created by calling environments, and PEPSKit provides some useful methods to access the boundary tensors.
# One minor caveat, environments() also normalizes the peps tensors behind the scenes.
# It's much easier to work with our algorithms if the peps is normalized...
env = environments(peps);
codomain(AC(env,North,2,2)) ← domain(AC(env,North,2,2))

# As you can see, by default the environment tensors are trivial chi = 1.
# There is no easy way to change this, at the moment you have to make use of the MPSKit finalize - changebonds machinery.

function bound_finalize(iter,state,ham,pars)
    maxD = 20;

    # check every virtual bond dimension and see if it's larger then maxD
    bigenough = reduce((a,(i,j))-> a && dim(virtualspace(state,i,j))>=maxD,
        Iterators.product(1:size(state,1),1:size(state,2)),init=true)

    if !bigenough
        (state,pars) = changebonds(state,ham,OptimalExpand(trscheme=truncdim(1)),pars)
    end

    return (state,pars)
end
env = environments(peps,alg=Vumps(verbose=false,finalize = bound_finalize))
codomain(AC(env,North,2,2)) ← domain(AC(env,North,2,2))

#%%
# It is now very easy to calculate the norm of our infinite mps in a bunch of different ways
@tensor fp1RL(env,North,1,1)[1,2,3,4]*fp1LR(env,South,0,1)[4,2,3,1]
@tensor fp0RL(env,East,1,1)[1,2]*fp0LR(env,West,0,2)[2,1]
@tensor fp1LR(env,North,2,2)[1,2,3,4]*
        corner(env,NorthEast,2,2)[4,5]*
        fp1LR(env,East,2,2)[5,6,7,8]*
        corner(env,SouthEast,2,2)[8,9]*
        fp1LR(env,South,2,2)[9,10,11,12]*
        corner(env,SouthWest,2,2)[12,13]*
        fp1LR(env,West,2,2)[13,14,15,16]*
        corner(env,NorthWest,2,2)[16,1]*
        peps[2,2][14,10,6,2,17]*
        conj(peps[2,2][15,11,7,3,17])

# Anyway, this probably makes very little sense unless you know the ins and outs already.
# It should clarify why things like measuring expectation values require passing in the environment object itself :
sz = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)
heatmap(real.(expectation_value(env,sz)))


ham = nonsym_nn_xxz_ham();
expectation_value(env,ham)

#%%

# To numerically approximate the gradient, we can use a number of different techniques.
# One such approximation is known as the channel approximation. To calculate these channels, you can simply call
chan = channels(env,ham);

# which can be passed to effectivehn
(H_eff,N_eff) = PEPSKit.effectivehn(chan,2,2)
codomain(H_eff) ← domain(H_eff)

# to calculate the gradient
v = permute(peps[2,2],(1,2,3,4,5))
gradient = H_eff*v - dot(v,H_eff*v)*N_eff*v
norm(gradient)

#%%
# The energy can also be minimized using algorithms from another package - OptimKit.
optalg = LBFGS(linesearch = HagerZhangLineSearch(ϵ = 1e-3),maxiter=500,verbosity=2,gradtol=1e-2)

(gs_state,gs_chan,delta) = find_groundstate(peps,ham,optalg,chan);
heatmap(real.(expectation_value(gs_chan,sz)))
