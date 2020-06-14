using MPSKit,PEPSKit,TensorKit,OptimKit,Plots

ham = nonsym_nn_xxz_ham();

peps = FinPEPS(map(Iterators.product(1:10,1:10)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',ℂ^2)
end);

pars = params(peps,ham,Dmrg2(trscheme=truncdim(10),verbose=false));
optalg = LBFGS(linesearch = HagerZhangLineSearch(ϵ = 1e-3,verbosity=0),maxiter=20,verbosity=20,gradtol=1e-4)
(peps,pars,delta) = find_groundstate(peps,ham,optalg,pars)

(sx,sy,sz,id) = nonsym_spintensors(1//2);
heatmap(real.(expectation_value(peps,sz,pars)),clim=(-0.5,0.5))
savefig("heatmap.png")
