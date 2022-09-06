using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit

p = InfinitePEPS(fill(ℂ^2,1,1),fill(ℂ^2,1,1));

ham = TensorMap(rand,ComplexF64,ℂ^2,ℂ^2);
ham += ham';

function cfun(p)
    env = leading_boundary(p,CTMRG(verbose=0,trscheme=truncdim(50),maxiter=10));
    real(sum(expectation_value(env,ham)))
end


for i in 1:10
    @show cfun(p)
    p.A.-=cfun'(p).A .* 0.01;
end

