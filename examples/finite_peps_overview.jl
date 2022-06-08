# Short overview of the finite peps code. It's recommended to first check out the infinite peps code.
using Revise,MPSKit,PEPSKit,TensorKit,OptimKit,Plots

#%%
#=
Finite peps is a simply a 2d array.
If the outward facing legs are not dim==1 then they are traced over and you're effectively minimizing over a density matrix.
=#

data = map(Iterators.product(1:5,1:5)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',(ℂ^2)')
end
peps = FinPEPS(data);

#%%
# You can construct the environment object, which contains the boundary finite mpses
env = environments(peps,DMRG2(trscheme=truncdim(50),verbose=false));

# which in turn can be used to measure observables
sz = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)
heatmap(real.(expectation_value(env,sz)))

ham = nonsym_nn_xxz_ham();
expectation_value(env,ham)

#%%

# The gradient can be approximated using the channels. This is pretty fast, but the precision is fundamentally limited.
chan = channels(env,ham);
optalg = LBFGS(linesearch = HagerZhangLineSearch(ϵ = 1e-3), maxiter=10,verbosity = 2,gradtol=1e-2)

(peps,chan,delta) = find_groundstate(peps,ham,optalg,chan);

sz = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)
heatmap(real.(expectation_value(chan,sz)))


#%%
# There is another way to find the gradient, which is cryptically called the correlator approach.
# It should be slower but becomes exact in the limit of chi going to infinity. It can be used by calling
cor = correlator(env,ham);
(peps,cor,delta) = find_groundstate(peps,ham,optalg,cor);
heatmap(real.(expectation_value(cor,sz)))
