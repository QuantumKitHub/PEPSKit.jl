function MPSKit.find_groundstate(peps::InfPEPS,ham::NN,alg::OptimKit.OptimizationAlgorithm;pars=params(peps,ham),bound_finalize =(iter,state,ham,pars)->(state,pars))
    #=
        - I need to clean this up
        - transition map was needed to define transport (and therefore used lbfgs,cg)
    =#

    #to call optimkit we will pack (peps,prevpars) together in a tuple
    #the gradient type will simply be a 2d array of tensors
    function objfun(x)
        (cpe,cpr,old_tm) = x;

        cg = map(Iterators.product(1:size(cpe,1),1:size(cpe,2))) do (i,j)
            (heff,neff) = effectivehn(cpr,i,j);
            v = permute(cpe[i,j],(1,2,3,4,5));
            permute(heff*v - dot(v,heff*v)*neff*v,(1,2,3,4),(5,))
        end

        real(expectation_value(cpe,ham,cpr))/(size(cpe,1)*size(cpe,2)),cg
    end

    function retract(x, cgr, α)
        (cpe,cpr,old_tm) = x;

        @info "trying stepsize $α"
        flush(stdout)

        #we on't want retract to overwrite the old state!
        npe = deepcopy(cpe)
        npr = deepcopy(cpr);
        for (i,j) in Iterators.product(1:size(cpe,1),1:size(cpe,2))
            @tensor npe[i,j][-1 -2 -3 -4;-5]+=(α*cgr[i,j])[-1,-2,-3,-4,-5]
        end

        prevnorms = map(norm,npe);
        MPSKit.recalculate!(npr,npe,bound_finalize=bound_finalize)
        newnorms = map(norm,npe);

        new_tm = copy(old_tm);
        newgrad = deepcopy(cgr);
        for (i,j) in Iterators.product(1:size(cpe,1),1:size(cpe,2))
            new_tm[i,j] = newnorms[i,j]/prevnorms[i,j];
            newgrad[i,j]*=new_tm[i,j];
        end


        #should also calculate "local gradient along that path"
        return (npe,npr,new_tm),newgrad
    end

    function inner(x, v1, v2)
        tot = 0.0;

        for (p1,p2) in zip(v1,v2)
            tot += 2*real(dot(v1,v2))
        end

        return tot
    end
    function transport!(v, xold, d, α, xnew)
        (_,_,tm) = xnew;
        for i in 1:size(v,1)
            for j in 1:size(v,2)
                v[i,j]*=tm[i,j]
            end
        end
        v
    end
    scale!(v, α) = v.*α
    add!(vdst, vsrc, α) = vdst+α.*vsrc


    (x,fx,gx,normgradhistory)=optimize(objfun,(peps,pars,ones(size(peps,1),size(peps,2))),alg;
        retract = retract,
        inner = inner,
        transport! = transport!,
        scale! = scale!,
        add! = add!,
        isometrictransport = false)

    return (x[1],x[2],normgradhistory[end])

    #return optimtest(objfun, (peps,pars), objfun((peps,pars))[2]; alpha= 0:0.01:0.1,retract = retract, inner = inner)
end


function MPSKit.find_groundstate(peps::A,ham::NN,alg::OptimKit.OptimizationAlgorithm,pars::B) where {A<:FinPEPS,B<:FinNNHamChannels}
    #to call optimkit we will pack (peps,prevpars) together in a tuple
    #the gradient type will simply be a 2d array of tensors
    function objfun(x::Tuple{A,B,Matrix{Float64}})
        (cpe,cpr,old_tm) = x;

        cg = map(Iterators.product(1:size(cpe,1),1:size(cpe,2))) do (i,j)
            (heff,neff) = effectivehn(cpr,i,j);
            v = permute(cpe[i,j],(1,2,3,4,5));
            n = dot(v,neff*v)
            permute(heff*v - (dot(v,heff*v)/n)*neff*v,(1,2,3,4),(5,))/n
        end

        #I don't know why I have to multiply, whereas I had to divide for infinite....
        en = real(expectation_value(cpe,ham,cpr))*(size(cpe,1)*size(cpe,2))
        en,cg
    end

    function retract(x::Tuple{A,B,Matrix{Float64}}, cgr, α)
        (cpe,cpr,old_tm) = x;

        @info "trying stepsize $α"
        flush(stdout)

        #we on't want retract to overwrite the old state!
        npe::A = deepcopy(cpe)
        npr::B = deepcopy(cpr);
        new_tm = ones(Float64,size(cpe,1),size(cpe,2))

        for (i,j) in Iterators.product(1:size(cpe,1),1:size(cpe,2))
            @tensor npe[i,j][-1 -2 -3 -4;-5]+=(α*cgr[i,j])[-1,-2,-3,-4,-5]

            newn = norm(npe[i,j])
            normalize!(npe[i,j])

            new_tm[i,j] /=newn
        end

        MPSKit.recalculate!(npr,npe)

        newgrad = copy(cgr);
        for (i,j) in Iterators.product(1:size(cpe,1),1:size(cpe,2))
            newgrad[i,j]*=new_tm[i,j];
        end

        #should also calculate "local gradient along that path"
        return (npe,npr,new_tm),newgrad
    end

    function inner(x::Tuple{A,B,Matrix{Float64}}, v1, v2)
        tot = 0.0;

        for (p1,p2) in zip(v1,v2)
            tot += real(dot(v1,v2))*2
        end

        return tot
    end
    function transport!(v, xold::Tuple{A,B,Matrix{Float64}}, d, α, xnew::Tuple{A,B,Matrix{Float64}})
        (_,_,tm) = xnew;
        for i in 1:size(v,1)
            for j in 1:size(v,2)
                v[i,j]*=tm[i,j]
            end
        end
        v
    end
    scale!(v, α) = v.*α
    add!(vdst, vsrc, α) = vdst+α.*vsrc

    (x,fx,gx,normgradhistory)=optimize(objfun,(peps,pars,ones(Float64,size(peps,1),size(peps,2))),alg;
        retract = retract,
        inner = inner,
        transport! = transport!,
        scale! = scale!,
        add! = add!,
        isometrictransport = false)

    return (x[1],x[2],normgradhistory[end])
    #return optimtest(objfun, (peps,pars,ones(Float64,size(peps,1),size(peps,2))), objfun((peps,pars,ones(Float64,size(peps,1),size(peps,2))))[2]; alpha= 0:0.01:0.1,retract = retract, inner = inner)
end
