function nonsym_nn_xxz_ham(;spin=1//2)
    (sx,sy,sz,id) = nonsym_spintensors(spin);
    @tensor nn[-1 -2;-3 -4]:=sx[-1,-2]*sx[-3,-4]+sy[-1,-2]*sy[-3,-4]+sz[-1,-2]*sz[-3,-4]
    return NN(nn)
end

function su2_nn_xxz_ham(;spin = 1//2)
    #only checked for spin = 1 and spin = 2...
    ph = ℂ[SU₂](spin=>1)

    a = TensorMap(ones, ComplexF64, ph , ℂ[SU₂](1=>1)*ph)
    b = TensorMap(ones, ComplexF64, ℂ[SU₂](1=>1)*ph , ph)
    @tensor ham[-1 -2;-3 -4]:=a[-1,1,-2]*b[1,-3,-4]
    NN(ham)
end

function u1_nn_xxz_ham(;spin = 1//2)
    (sxd,syd,szd,idd) = MPSKit.spinmatrices(spin);
    @tensor ham[-1 -2;-3 -4]:=sxd[-1,-3]*sxd[-2,-4]+syd[-1,-3]*syd[-2,-4]+szd[-1,-3]*szd[-2,-4]

    indu1map = [U₁(v) for v in -spin:1:spin];
    pspace = U1Space((v=>1 for v in indu1map));

    symham = TensorMap(zeros,eltype(ham),pspace*pspace,pspace*pspace)

    for (i,j,k,l) in Iterators.product(1:size(ham,1),1:size(ham,1),1:size(ham,1),1:size(ham,1))
        if ham[i,j,k,l]!=0
            copyto!(symham[(indu1map[i],indu1map[j],indu1map[k],indu1map[l])],ham[i,j,k,l])
        end
    end

    NN(permute(symham,(1,3),(2,4)))
end
