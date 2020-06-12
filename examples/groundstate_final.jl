using MPSKit,PEPSKit,TensorKit,OptimKit

ham = nonsym_nn_xxz_ham();

peps = FinPEPS(map(Iterators.product(1:10,1:10)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',ℂ^2)
end);

pars = params(peps,ham,Dmrg2(trscheme=truncdim(10),verbose=false));
optalg = LBFGS(linesearch = HagerZhangLineSearch(ϵ = 1e-3,verbosity=0),maxiter=500,verbosity=20,gradtol=1e-4)
(gspeps,gspars,delta) = find_groundstate(peps,ham,optalg,pars)
