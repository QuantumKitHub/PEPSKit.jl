using MPSKit,PEPSKit,TensorKit,OptimKit

ham = nonsym_nn_ising_ham();

# 2 by 2 unit cell
peps = InfPEPS(map(Iterators.product(1:2,1:2)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^2*ℂ^2*(ℂ^2)'*(ℂ^2)',ℂ^2)
end);

(peps,pars,_) = find_groundstate(peps,ham,ConjugateGradient(linesearch = HagerZhangLineSearch(ϵ = 1e-6),maxiter=50,verbosity=2,gradtol=1e-5))

@show expectation_value(peps,ham,pars)
