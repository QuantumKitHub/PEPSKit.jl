#I don't know if we should move this to mpskit somehow, or leave it here

function approximate(init,pepsline,state,alg::Dmrg2,pars=params(init,pepsline,state))
    tol=alg.tol;maxiter=alg.maxiter
    iter = 0; delta = 2*tol

    while iter < maxiter && delta > tol
        delta=0.0

        for pos=[1:(length(state)-1);length(state)-2:-1:1]

            newA2center = downproject2(pos,init,pepsline,state,pars)

            (al,c,ar) = tsvd(newA2center,trunc=alg.trscheme)

            #yeah, we need a different convergence criterium
            olda2c = MPSKit._permute_front(init.AL[pos])*init.CR[pos]*MPSKit._permute_tail(init.AR[pos+1])

            init.AC[pos] = (al,complex(c))
            init.AR[pos+1] = MPSKit._permute_front(ar);

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
