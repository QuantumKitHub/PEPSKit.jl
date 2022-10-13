using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit

#A = TensorMap(rand,ComplexF64,ℂ^1,ℂ^2);
#@otensor A[1,2]*A[2,1]
#p = TensorMap(rand,ComplexF64,ℂ^1⊗(ℂ^1)',ℂ^2⊗ℂ^2⊗(ℂ^2)'⊗(ℂ^2)');
p = InfinitePEPS([ℂ^1 ℂ^1;ℂ^1 ℂ^1],[ℂ^2 ℂ^2;ℂ^2 ℂ^2],[ℂ^2 ℂ^2;ℂ^2 ℂ^2]);
p = InfinitePEPS(reshape([ℂ^1],1,1),reshape([ℂ^2],1,1),reshape([ℂ^2],1,1));
penvs =  leading_boundary(p,CTMRG(verbose=1,trscheme=truncdim(20)));
#onsitenorm(penvs)
#ham = LocalHamiltonian(repeat(PeriodicArray([p]),1,1));