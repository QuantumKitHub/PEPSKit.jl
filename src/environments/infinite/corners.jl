#=
    Can interpret vumps output in the pulling-through kind of way
    This gives 2 equivalent ways of obtaining corner matrices
        - lfps
        - rfps

    I then kinda take the average of the two, and use that
=#
function northwest_corner_tensors!(init,nbound,npars,wbound,wpars,peps;verbose=true)
    #cornerprime
    (nrows,ncols) = size(peps)

    lfps = similar(init)
    for i in 1:nrows
        curl = [rightenv(wpars,s,nrows-i+1,wbound) for s in 1:ncols];
        botl = nbound.AL[i,1:ncols];

        if (space(init[i,1],1) != space(botl[1],1)) || (space(init[i,1],2)' != space(curl[1],1))
            #check if init is actually compatible
            #if not; initialize with a random one
            init[i,1] = TensorMap(rand,ComplexF64,space(botl[1],1),space(curl[1],1))
        end

        (vals,vecs,convhist)=eigsolve(x->transfer_left(x,curl,botl),init[i,1],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "lcorner failed to converge"
        lfps[i,1] = vecs[1]

        for j in 2:ncols
            lfps[i,j] = transfer_left(lfps[i,j-1],curl[j-1],botl[j-1])
        end
    end

    rfps = similar(init)
    for i in 1:ncols
        curr = [leftenv(npars,1-s,i,nbound) for s in 1:nrows];
        botr = wbound.AR[i,1:nrows];

        (vals,vecs,convhist)=eigsolve(x->transfer_right(x,curr,botr),lfps[1,i],1,:LM,Arnoldi());
        convhist.converged == 0 && @info "rcorner failed to converge"
        rfps[1,i] = vecs[1]

        for j in 2:nrows
            rfps[j,i] = transfer_right(rfps[j-1,i],curr[end-j+2],botr[end-j+2])
        end
    end

    for i in 1:size(init,1), j in 1:size(init,2)
        rmul!(rfps[i,j],dot(rfps[i,j],lfps[i,j]));

        normalize!(rfps[i,j]);
        normalize(lfps[i,j]);

        verbose && @info "corner inconsistency $(norm(l-r))"

        init[i,j] = 0.5*(lfps[i,j]+rfps[i,j])
    end

    return init
end
