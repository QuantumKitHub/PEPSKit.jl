#=
    Given a peps, we would like access to
        - boundary mps's
        - corner tensors (channel approximation)
        - precomputed fixpoints
=#
mutable struct InfEnvManager{T,A,B,C,D} <: Cache
    peps::T
    boundaries::A
    corners::B

    fp0::C
    fp1::D
    #fp2::E
end

function MPSKit.params(peps::InfPEPS;kwargs...)
    #=
        All we do here is create a trivial environment
        and then pass it to recalculate, which takes an old environment and calculates the new one
    =#

    ou = oneunit(space(peps,1,1));

    #generate trivial boundaries
    boundaries = PeriodicArray(map(Dirs) do dir

        bs = map(rotate_north(peps,dir)) do p
            TensorMap(rand,ComplexF64,ou*space(p,North)'*space(p,North),ou)
        end

        MPSMultiline(bs)
    end)

    #generate trivial corners
    corners = PeriodicArray(map(Dirs) do dir
        (numrows,numcols) = size(boundaries[dir]);

        PeriodicArray(map(Iterators.product(1:numrows,1:numcols)) do (i,j)
            TensorMap(rand,ComplexF64,ou,ou)
        end)
    end)

    #generate trivial fp0
    fp0 = PeriodicArray(map(Dirs) do dir
        (numrows,numcols) = size(boundaries[dir]);

        PeriodicArray(map(Iterators.product(1:numrows,1:numcols)) do (i,j)
            TensorMap(rand,ComplexF64,ou,ou)
        end)
    end)

    #generate trivial fp1
    fp1 = PeriodicArray(map(Dirs) do dir
        (numrows,numcols) = size(boundaries[dir]);

        PeriodicArray(map(Iterators.product(1:numrows,1:numcols)) do (i,j)
            pspace = space(boundaries[dir].AL[i,j],2)
            TensorMap(rand,ComplexF64,ou*pspace*pspace',ou)
        end)
    end)

    #generate trivial fp2
    #=
    fp2 = PeriodicArray(map(Dirs) do dir
        (numrows,numcols) = size(boundaries[dir]);

        PeriodicArray(map(Iterators.product(1:numrows,1:numcols)) do (i,j)
            pspace1 = space(boundaries[dir].AL[i,j],2)
            pspace2 = space(boundaries[dir].AL[i,j+1],2)
            TensorMap(rand,ComplexF64,ou*pspace1*pspace1'*pspace2*pspace2',ou)
        end)
    end)
    =#
    return MPSKit.recalculate!(InfEnvManager(peps,boundaries,corners,fp0,fp1#=,fp2=#),peps;kwargs...)
end

function MPSKit.recalculate!(prevenv::InfEnvManager,peps::InfPEPS;verbose = false,tol = 1e-10,bound_finalize = (iter,state,ham,pars)->(state,pars),maxiter=1000)
    prevenv.peps = peps;

    #pars == the boundary mps parameters
    boundpars = map(Dirs) do dir
        @Threads.spawn north_boundary_mps(rotate_north(peps,dir),prevenv.boundaries[dir],verbose=verbose,tol=tol,bound_finalize=bound_finalize,maxiter=maxiter);
    end

    pars = map(Dirs) do dir
        (prevenv.boundaries[dir],par,err) = fetch(boundpars[dir])
        par
    end

    for dir in Dirs
        prevenv.corners[dir] = northwest_corner_tensors(prevenv.corners[dir],
            prevenv.boundaries[dir],pars[dir],
            prevenv.boundaries[left(dir)],pars[left(dir)],
            rotate_north(peps,dir),verbose=verbose);
    end

    #determines 0-size channel fixpoints and fixes boundary mps phases
    (prevenv.fp0[West],prevenv.fp0[East]) = fp0!(prevenv.boundaries[North],prevenv.boundaries[South],verbose=verbose)
    (prevenv.fp0[North],prevenv.fp0[South]) = fp0!(prevenv.boundaries[East],prevenv.boundaries[West],verbose=verbose)
    #determine 1 and 2 size channel fixpoints

    fps = map(Dirs) do dir
        fp1 = @Threads.spawn north_fp1(prevenv.boundaries[left(dir)],rotate_north(peps,dir),prevenv.boundaries[right(dir)],verbose=verbose);
        #fp2 = north_fp2(prevenv.boundaries[left(dir)],rotate_north(peps,dir),prevenv.boundaries[right(dir)],verbose=verbose);
        (fp1,#=fp2=#)
    end

    for dir in Dirs
        prevenv.fp1[dir] = fetch(fps[dir][1]);
        #prevenv.fp2[dir] = fps[dir][2];
    end

    renormalize!(prevenv,verbose=verbose)

    return prevenv
end

function renormalize!(man::InfEnvManager;verbose=false)

    #=
        In this bit of code we try to fix all free parameters
        such that the consistency equations are all mostly valid
    =#

    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))

        nw = man.corners[NorthWest][i,j];
        ne = man.corners[NorthEast][end-j+2,i];
        se = man.corners[SouthEast][end-i+2,end-j+2];
        sw = man.corners[SouthWest][j,end-i+2];

        n0 = man.fp0[North][i,j];
        e0 = man.fp0[East][end-j+2,i];
        s0 = man.fp0[South][end-i+2,end-j+2];
        w0 = man.fp0[West][j,end-i+2];

        n = man.boundaries[North].CR[i,j-1];
        e = man.boundaries[East].CR[end-j+2,i-1];
        s = man.boundaries[South].CR[end-i+2,end-j+1];
        w = man.boundaries[West].CR[j,end-i+1];

        a = dot(n,nw*n0*ne);
        b = dot(e,ne*e0*se);
        c = dot(s,se*s0*sw);
        d = dot(w,sw*w0*nw);

        #an approximate solution for the problem is given by:
        rmul!(ne,1/a);
        rmul!(se,d/c)
        rmul!(sw,1/d)

        # a*c should equal b*d - we are not able to do change that
        # assuming this is equal, the consistency equation can be fully solved
        verbose && println("fp0 inconsistency $(abs(a*c-b*d))")

        #these are all the consistency equations we impose in this step : ...
        verbose && println("fp0 - nw inconsistency $(norm(nw*n0*ne-n))")
        verbose && println("fp0 - ne inconsistency $(norm(ne*e0*se-e))")
        verbose && println("fp0 - se inconsistency $(norm(se*s0*sw-s))")
        verbose && println("fp0 - sw inconsistency $(norm(sw*w0*nw-w))")
    end

    for dir in Dirs
        nw = man.corners[dir];ne = man.corners[right(dir)];n = man.boundaries[dir]
        n1 = man.fp1[dir];#n2 = man.fp2[dir];

        (tnr,tnc) = rotate_north(size(man.peps),dir)
        for (i,j) in Iterators.product(1:tnr,1:tnc)
            @tensor tout[-1 -2 -3;-4]:=nw[i,j][-1,2]*n1[i,j][2,-2,-3,3]*ne[end-j+1,i][3,-4]
            val = dot(tout,n.AC[i,j])/(norm(tout)*norm(tout));
            rmul!(n1[i,j],val)

            @tensor tout[-1 -2 -3;-4]:=nw[i,j][-1,2]*n1[i,j][2,-2,-3,3]*ne[end-j+1,i][3,-4]
            verbose && println("fp1 inconsistency $(norm(tout-n.AC[i,j]))");

            #=
            @tensor tout[-1 -2 -3 -4 -5;-6]:=nw[i,j][-1,2]*n2[i,j][2,-2,-3,-4,-5,3]*ne[end-j,i][3,-6]
            @tensor shouldbe[-1 -2 -3 -4 -5;-6]:=n.AC[i,j][-1,-2,-3,1]*n.AR[i,j+1][1,-4,-5,-6]
            val = dot(tout,shouldbe)/(norm(tout)*norm(tout));
            rmul!(n2[i,j],val)

            @tensor tout[-1 -2 -3 -4 -5;-6]:=nw[i,j][-1,2]*n2[i,j][2,-2,-3,-4,-5,3]*ne[end-j,i][3,-6]
            verbose && println("fp2 inconsistency $(norm(tout-shouldbe))");
            =#
        end
    end

    #renormalize the peps itself
    for (i,j) in Iterators.product(1:size(man.peps,1),1:size(man.peps,2))
        # Dirs = West,South,East,North
        (w,s,e,n) = map(Dirs) do dir
            man.fp1[dir][rotate_north((i,j),size(man.peps),dir)...]
        end

        (sw,se,ne,nw) = map(Dirs) do dir
            man.corners[dir][rotate_north((i,j),size(man.peps),dir)...]
        end

        ans = @tensor n[1,2,3,4]*ne[4,5]*e[5,6,7,8]*se[8,9]*s[9,10,11,12]*sw[12,13]*w[13,14,15,16]*nw[16,1]*
        man.peps[i,j][14,10,6,2,17]*conj(man.peps[i,j][15,11,7,3,17])

        verbose && println("localnorm was |$(ans)| = $(abs(ans))")
        rmul!(man.peps[i,j],1/sqrt(abs(ans)))
    end

end

Base.rotl90(envm::InfEnvManager) = InfEnvManager(   rotl90(envm.peps),
                                                    circshift(envm.boundaries,1),
                                                    circshift(envm.corners,1),
                                                    circshift(envm.fp0,1),
                                                    circshift(envm.fp1,1),
                                                    #=circshift(envm.fp2,1)=#)
Base.rotr90(envm::InfEnvManager) = InfEnvManager(   rotr90(envm.peps),
                                                    circshift(envm.boundaries,-1),
                                                    circshift(envm.corners,-1),
                                                    circshift(envm.fp0,-1),
                                                    circshift(envm.fp1,-1),
                                                    #=circshift(envm.fp2,-1)=#)
