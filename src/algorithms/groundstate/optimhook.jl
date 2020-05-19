function MPSKit.find_groundstate(peps::InfPEPS,ham::NN,alg::OptimKit.OptimizationAlgorithm;pars=params(peps,ham),finalize! =(x,f,g,numiter)->(x,f,g))
    #to call optimkit we will pack (peps,prevpars) together in a tuple
    #the gradient type will simply be a 2d array of tensors
    function objfun(x)
        (cpe,cpr) = x;

        cg = map(Iterators.product(1:size(cpe,1),1:size(cpe,2))) do (i,j)
            (heff,neff) = effectivehn(cpr,i,j);
            v = permute(cpe[i,j],(1,2,3,4,5));
            permute(heff*v - dot(v,heff*v)*neff*v,(1,2,3,4),(5,))
        end

        real(expectation_value(cpe,ham,cpr)),cg
    end

    function retract(x, cgr, α)
        (cpe,cpr) = x;

        #we on't want retract to overwrite the old state!
        npe = deepcopy(cpe)
        npr = deepcopy(cpr);

        for (i,j) in Iterators.product(1:size(cpe,1),1:size(cpe,2))
            @tensor npe[i,j][-1 -2 -3 -4;-5]+=(α*cgr[i,j])[-1,-2,-3,-4,-5]
            npe[i,j]=npe[i,j]/norm(npe[i,j])
        end
        MPSKit.recalculate!(npr,npe)

        #should also calculate "local gradient along that path"
        return (npe,npr),cgr
    end

    function inner(x, v1, v2)
        tot = 0.0;

        for (p1,p2) in zip(v1,v2)
            tot += 2*real(dot(v1,v2))
        end

        return tot
    end
    transport!(v, xold, d, α, xnew) = v
    scale!(v, α) = v.*α
    add!(vdst, vsrc, α) = vdst+α.*vsrc

    (x,fx,gx,normgradhistory)=optimize(objfun,(peps,pars),alg;
        retract = retract,
        inner = inner,
        transport! = transport!,
        scale! = scale!,
        add! = add!,
        finalize! = finalize!,
        isometrictransport = false)

    return (x[1],x[2],normgradhistory[end])
end
