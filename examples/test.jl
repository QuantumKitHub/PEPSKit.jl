using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit

p = InfinitePEPS(fill(ℂ^2,2,2),fill(ℂ^2,2,2));

ham = TensorMap(rand,ComplexF64,ℂ^2,ℂ^2);
ham += ham';

function cfun(p)
    env = leading_boundary(p,CTMRG(trscheme=truncdim(5)));
    real(sum(expectation_value(env,ham)))
end

cfun(p)


cfun'(p)