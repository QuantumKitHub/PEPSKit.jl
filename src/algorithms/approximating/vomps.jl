#I don't know if we should move this to mpskit somehow, or leave it here

function approximate(init::Union{MPSComoving,FiniteMPS},sq::Tuple,alg,pars=params(init,sq))
    tor =  approximate(init,[sq],alg,[pars]);
    return (tor[1],tor[2][1],tor[3])
end

function approximate(init::Union{MPSComoving,FiniteMPS},squash::Vector,alg::Dmrg2,pars=[params(init,sq) for sq in squash])

    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        for pos=[1:(length(init)-1);length(init)-2:-1:1]

            newA2center = sum(map(zip(squash,pars)) do (sq,pr)
                downproject2(pos,init,sq,pr)
            end)

            (al,c,ar) = tsvd(newA2center,trunc=alg.trscheme)

            #yeah, we need a different convergence criterium
            olda2c = MPSKit._permute_front(init.AL[pos])*init.CR[pos]*MPSKit._permute_tail(init.AR[pos+1])

            init.AC[pos] = (al,complex(c))
            init.AC[pos+1] = (complex(c),MPSKit._permute_front(ar));

            newa2c = MPSKit._permute_front(init.AL[pos])*init.CR[pos]*MPSKit._permute_tail(init.AR[pos+1])
            delta = max(delta,norm(olda2c-newa2c)/(1e-14+norm(newa2c)));
        end

        alg.verbose && @show (iter,delta)
        flush(stdout)
        #finalize
        iter += 1
    end

    return init,pars,delta
end

function approximate(init::Union{MPSComoving,FiniteMPS}, squash::Vector,alg::Dmrg,pars = [params(init,sq) for sq in squash])

    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        #finalize
        (init,pars) = alg.finalize(iter,init,squash,pars);

        for pos = [1:(length(init)-1);length(init):-1:2]
            newac = sum(map(zip(squash,pars)) do (sq,pr)
                downproject(pos,init,sq,pr)
            end)

            delta = max(delta,norm(newac-init.AC[pos])/norm(newac))

            init.AC[pos] = newac
        end

        alg.verbose && @show (iter,delta)
        flush(stdout)

        iter += 1
    end

    return init,pars,delta
end
