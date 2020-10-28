function MPSKit.find_groundstate(peps::InfPEPS,ham::NN,alg::OptimKit.OptimizationAlgorithm,pars::InfNNHamChannels)
    function objfun(x)
        (cpe,cpr,old_tm) = x;

        cg = map(product(1:size(cpe,1),1:size(cpe,2))) do (i,j)
            (heff,neff) = effectivehn(cpr,i,j);
            v = permute(cpe[i,j],(1,2,3,4,5));
            permute(heff*v - dot(v,heff*v)*neff*v,(1,2,3,4),(5,))
        end

        real(expectation_value(cpr.envm,ham))/(size(cpe,1)*size(cpe,2)),cg
    end

    function retract(x, cgr, α)
        (cpe,cpr,old_tm) = x;

        @info "trying stepsize $α"
        flush(stdout)

        #we don't want retract to overwrite the old state!
        new_peps = copy(cpe);
        new_pars = deepcopy(cpr);

        for i in 1:size(cpe,1), j in 1:size(cpe,2)
            new_peps[i,j] += α*cgr[i,j]
        end

        prevnorms = map(norm,new_peps);
        recalculate!(new_pars,new_peps)
        newnorms = map(norm,new_peps);

        new_tm = newnorms./prevnorms;
        newgrad = copy(cgr);
        for i in 1:size(cpe,1), j in 1:size(cpe,2)
            newgrad[i,j]*=new_tm[i,j];
        end

        #should also calculate "local gradient along that path"
        return (new_peps,new_pars,new_tm),newgrad
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
        for i in 1:size(v,1), j in 1:size(v,2)
            v[i,j]*=tm[i,j]
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


function MPSKit.find_groundstate(peps::A,ham::NN,alg::OptimKit.OptimizationAlgorithm,envs::B) where {A<:Union{WinPEPS,FinPEPS},B<:Union{FinNNHamCors,WinNNHamChannels,FinNNHamChannels}}
    #=
    we will rescale the peps tensors to make them uniformly gauged
    this will change the gradients, so we need to keep track of those rescale operations
    we will not pass along the environments, and instead update them in-place when objfun is called
    =#

    function objfun(x::Tuple{A,Matrix{Float64}})
        (peps,old_tm) = x;
        recalculate!(envs,peps)

        en = 0.0;
        cg = map(product(1:size(peps,1),1:size(peps,2))) do (i,j)
            (heff,neff) = effectivehn(envs,i,j);
            v = permute(peps[i,j],(1,2,3,4,5));

            n = dot(v,neff*v)
            h = dot(v,heff*v)
            en += real(h/n)

            permute(heff*v - (h/n)*neff*v,(1,2,3,4),(5,))/n
        end
        en,cg
    end

    function retract(x::Tuple{A,Matrix{Float64}}, cgr, α)
        (peps,old_tm) = x;

        @info "trying stepsize $α"
        flush(stdout)

        #we don't want retract to overwrite the old state!
        new_peps::A = copy(peps);
        new_grad = copy(cgr);
        new_tm = ones(Float64,size(peps,1),size(peps,2))

        for (i,j) in product(1:size(peps,1),1:size(peps,2))
            new_peps[i,j] += α*cgr[i,j]

            newn = norm(new_peps[i,j])

            normalize!(new_peps[i,j])
            new_grad[i,j] /= newn
            new_tm[i,j] /= newn
        end


        return (new_peps,new_tm),new_grad
    end

    function inner(x::Tuple{A,Matrix{Float64}}, v1, v2)
        tot = 0.0;

        for (p1,p2) in zip(v1,v2)
            tot += real(dot(v1,v2))*2
        end

        return tot
    end
    function transport!(v, xold::Tuple{A,Matrix{Float64}}, d, α, xnew::Tuple{A,Matrix{Float64}})
        (_,tm) = xnew;
        for i in 1:size(v,1), j in 1:size(v,2)
            v[i,j]*=tm[i,j]
        end
        v
    end
    scale!(v, α) = v.*α
    add!(vdst, vsrc, α) = vdst+α.*vsrc
    #return optimtest(objfun, (peps,ones(Float64,size(peps,1),size(peps,2))), objfun((peps,ones(Float64,size(peps,1),size(peps,2))))[2]; alpha= 0:0.1:0.2,retract = retract, inner = inner)

    (x,fx,gx,normgradhistory)=optimize(objfun,(peps,ones(Float64,size(peps,1),size(peps,2))),alg;
        retract = retract,
        inner = inner,
        transport! = transport!,
        scale! = scale!,
        add! = add!,
        isometrictransport = false)

    return (x[1],envs,normgradhistory[end])

end
