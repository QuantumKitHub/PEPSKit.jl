#=
    This method is a bit deceptive. Not only do we find the fp0 fixpoints,
    but we also immediatly change the boundary mpses to ensure a |1| norm
    maybe we should do that later on in renormalize!
    but then that would also complicate the logic there ...
=#
function fp0!(east,west;verbose=false)
    (ncols,nrows) = size(east);

    nfps = PeriodicArray{typeof(east.CR[1]),2}(undef,nrows,ncols);
    sfps = PeriodicArray{typeof(east.CR[1]),2}(undef,nrows,ncols);

    for i in 1:ncols
        initr = TensorMap(rand,ComplexF64,space(east.AL[end-i+2,end],4)',space(west.AL[i,1],1))
        initl = TensorMap(rand,ComplexF64,space(west.AL[i,end],4)',space(east.AL[end-i+2,1],1))

        (lva,lve,convhist) = eigsolve(x->crosstransfer(x,east.AL[end-i+2,:],reverse(west.AR[i,:])),initl,1,:LM,Arnoldi());
        convhist.converged == 0 && @info "lfp0 failed to converge"
        (rva,rve,convhist) = eigsolve(x->crosstransfer(x,west.AL[i,:],reverse(east.AR[end-i+2,:])),initr,1,:LM,Arnoldi());
        convhist.converged == 0 && @info "rfp0 failed to converge"

        verbose && println("leading lfp0 val $((lva[1]))")
        verbose && println("leading rfp0 val $((rva[1]))")
        @assert rva[1] ≈ lva[1]


        pref = (1.0/lva[1])^(1/(2*nrows));

        #first we change the phase of up.AL to make lva real
        for temp in 1:nrows
            rmul!(east.AL[end-i+2,temp],pref)
            rmul!(east.AC[end-i+2,temp],pref)
            rmul!(east.AR[end-i+2,temp],pref)

            rmul!(west.AL[i,temp],pref)
            rmul!(west.AC[i,temp],pref)
            rmul!(west.AR[i,temp],pref)
        end

        #=
            We already imposed that transferring over one unit cell has eigenvalue 1
            We also need that 2 contracted fixpoints have norm 1
        =#
        val = @tensor lve[1][1,2]*east.CR[end-i+2,0][2,3]*rve[1][3,4]*west.CR[i,end][4,1];
        nfps[1,i] = lve[1]/sqrt(val);
        sfps[1,end-i+2] = rve[1]/sqrt(val);

        #the other fixpoints are determined by doing the transfer
        for j in 2:nrows
            nfps[j,i] = crosstransfer(nfps[j-1,i],east.AL[end-i+2,j-1],west.AR[i,end-j+2])
            sfps[j,end-i+2] = crosstransfer(sfps[j-1,end-i+2],west.AL[i,j-1],east.AR[end-i+2,end-j+2])
        end

        for j in 1:nrows
            val = @tensor nfps[j,i][1,2]*east.CR[end-i+2,j-1][2,3]*sfps[end-j+2,end-i+2][3,4]*west.CR[i,end-j+1][4,1];
            verbose && println("fp0 inconsistency $(abs(val-1))")
        end

    end

    return nfps,sfps
end

#---- I don't know how to clean the following up

#gets leading fix points in the north direction
function north_fp1(west,peps,east;verbose = false)

    (nrows,ncols) = size(peps);

    nfps = PeriodicArray{typeof(west.AL[1]),2}(undef,nrows,ncols);

    for i in 1:ncols
        initl = TensorMap(rand,ComplexF64,space(west.AL[i,nrows],4)'*space(peps[1,i],North)'*space(peps[1,i],North),space(east.AL[end-i+1,1],1))

        (lva,lve,convhist) = eigsolve(x->crosstransfer(x,peps[:,i],east.AL[end-i+1,:],reverse(west.AR[i,:])),initl,1,:LM,Arnoldi());
        convhist.converged == 0 && @info "fp1 failed to converge"
        verbose && println("leading fp1 val $(lva[1]))")

        nfps[1,i] = lve[1];

        for j in 2:nrows
            nfps[j,i] = crosstransfer(nfps[j-1,i],peps[j-1,i],east.AL[end-i+1,j-1],west.AR[i,nrows-j+2])
        end
    end

    return nfps
end

function north_fp2(west,peps,east;verbose = false)
    (nrows,ncols) = size(peps);

    nfps = Array{Any,2}(undef,nrows,ncols);

    for i in 1:ncols
        initl = TensorMap(rand,ComplexF64,space(west.AL[i,nrows],4)'*space(peps[1,i],North)'*space(peps[1,i],North)*space(peps[1,i+1],North)'*space(peps[1,i+1],North),space(east.AL[end-i,1],1))

        (lva,lve,convhist) = eigsolve(x->crosstransfer(x,peps[:,i],peps[:,i+1],east.AL[end-i,:],reverse(west.AR[i,:])),initl,1,:LM,Arnoldi());
        convhist.converged == 0 && @info "fp2 failed to converge"
        verbose && println("leading fp2 val $(lva[1])")
        nfps[1,i] = lve[1];

        for j in 2:nrows
            nfps[j,i] = crosstransfer(nfps[j-1,i],peps[j-1,i],peps[j-1,i+1],east.AL[end-i,j-1],west.AR[i,nrows-j+2])
        end
    end

    return PeriodicArray(convert(Array{typeof(nfps[1,1]),2},nfps))
end
