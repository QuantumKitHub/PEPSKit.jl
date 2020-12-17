# dot is a property of the peps itself, but requires numerical approximation.
# I use the same one as was used by the environment....
LinearAlgebra.dot(st1::InfEnvManager,st2::InfEnvManager) = dot(st1.peps,st2.peps,st1.alg)

function LinearAlgebra.dot(st1::InfPEPS,st2::InfPEPS,alg::MPSKit.Algorithm = Vumps(verbose=false))
    ou = oneunit(space(st1[1,1],1));

    #construct initial boundary mps

    bmps = MPSMultiline(reshape(map(zip(st1,st2)) do (p1,p2)
        TensorMap(rand,ComplexF64,ou*space(p2,North)'*space(p1,North),ou)
                end,size(st1)));

    (bmps,pars,_) = leading_boundary(bmps,(st2,st1),alg);

    total = map(product(1:size(st1,1),1:size(st1,2))) do (i,j)
        dot(bmps.AC[i+1,j],ac_prime(bmps.AC[i,j],i,j,bmps,pars))
    end

    sum(total)/length(total)
end
