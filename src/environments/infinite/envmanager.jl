#=
    Given a peps, we would like access to
        - boundary mps's
        - corner tensors (channel approximation)
        - precomputed fixpoints
=#
mutable struct InfEnvManager{T,A,B,C,D,E} <: Cache
    peps::T
    boundaries::A
    corners::B

    fp0::C
    fp1::D
    fp2::E
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
    fp2 = PeriodicArray(map(Dirs) do dir
        (numrows,numcols) = size(boundaries[dir]);

        PeriodicArray(map(Iterators.product(1:numrows,1:numcols)) do (i,j)
            pspace1 = space(boundaries[dir].AL[i,j],2)
            pspace2 = space(boundaries[dir].AL[i,j+1],2)
            TensorMap(rand,ComplexF64,ou*pspace1*pspace1'*pspace2*pspace2',ou)
        end)
    end)

    return MPSKit.recalculate!(InfEnvManager(peps,boundaries,corners,fp0,fp1,fp2),peps;kwargs...)
end

function MPSKit.recalculate!(prevenv::InfEnvManager,peps::InfPEPS;verbose = false,tol = 1e-10,bondmanage=SimpleManager())
    prevenv.peps = peps;

    #pars == the boundary mps parameters
    pars = map(Dirs) do dir
        (prevenv.boundaries[dir],par,err) = north_boundary_mps(rotate_north(peps,dir),prevenv.boundaries[dir],verbose=verbose,tol=tol,bondmanage=bondmanage);
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
    for dir in Dirs
        prevenv.fp1[dir] = north_fp1(prevenv.boundaries[left(dir)],rotate_north(peps,dir),prevenv.boundaries[right(dir)],verbose=verbose);
        prevenv.fp2[dir] = north_fp2(prevenv.boundaries[left(dir)],rotate_north(peps,dir),prevenv.boundaries[right(dir)],verbose=verbose);
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

        n = man.boundaries[North].CR[i,j-1];
        e = man.boundaries[East].CR[end-j+2,i-1];
        s = man.boundaries[South].CR[end-i+2,end-j+1];
        w = man.boundaries[West].CR[j,end-i+1];

        a = tr(nw*ne*se*sw);
        b = tr(n*e*s*w);

        val = b/a;
        val /= abs(val);

        rmul!(nw,val); # this has to hold - otherwise the norm won't be real
    end


    #solve fp0 inconcistency (we fix some bounary phase things as well in the meantime)
    for dir in [East,North]
        nw = man.corners[dir];ne = man.corners[right(dir)];
        n = man.boundaries[dir];
        e = man.boundaries[right(dir)];
        w = man.boundaries[left(dir)];
        n0 = man.fp0[dir];
        s0 = man.fp0[right(right(dir))];

        (tnr,tnc) = rotate_north(size(man.peps),dir)
        for (i,j) in Iterators.product(1:tnr,1:tnc)
            val = dot(n.CR[i,j-1],nw[i,j]*n0[i,j]*ne[end-j+2,i])
            val /= abs(val);
            rmul!(n0[i,j],1/val)
            rmul!(s0[end-i,end-j+2],val)

            rmul!(e.AL[end-j+2,i],val);
            rmul!(e.AR[end-j+2,i],val);
            rmul!(e.AC[end-j+2,i],val);
            rmul!(e.AL[end-j+2,i-1],1/val);
            rmul!(e.AR[end-j+2,i-1],1/val);
            rmul!(e.AC[end-j+2,i-1],1/val);;
        end
    end

    for dir in Dirs
        nw = man.corners[dir];ne = man.corners[right(dir)];n = man.boundaries[dir]
        n1 = man.fp1[dir];n2 = man.fp2[dir];

        (tnr,tnc) = rotate_north(size(man.peps),dir)
        for (i,j) in Iterators.product(1:tnr,1:tnc)
            #solve fp1 inconsistency
            val = @tensor conj(n.AC[i,j][-1,-2,-3,-4])*nw[i,j][-1,2]*n1[i,j][2,-2,-3,3]*ne[end-j+1,i][3,-4]
            rmul!(n1[i,j],1/val)

            #show remaining inconsistency
            @tensor tout[-1 -2 -3;-4]:=nw[i,j][-1,2]*n1[i,j][2,-2,-3,3]*ne[end-j+1,i][3,-4]
            verbose && println("fp1 inconsistency $(norm(tout-n.AC[i,j]))");

            #solve fp2 inconsistency
            val = @tensor conj(n.AC[i,j][-1,-2,-3,4])*conj(n.AR[i,j+1][4,-4,-5,-6])*nw[i,j][-1,2]*n2[i,j][2,-2,-3,-4,-5,3]*ne[end-j,i][3,-6]
            rmul!(n2[i,j],1/val)

            #show remaining inconsistency
            @tensor tout[-1 -2 -3; -4 -5 -6]:=nw[i,j][-1,2]*n2[i,j][2,-2,-3,-4,-5,3]*ne[end-j,i][3,-6]
            shouldbe = n.AC[i,j]*MPSKit._permute_tail(n.AR[i,j+1])
            verbose && println("fp2 inconsistency $(norm(tout-shouldbe))");
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
                                                    circshift(envm.fp2,1))
Base.rotr90(envm::InfEnvManager) = InfEnvManager(   rotr90(envm.peps),
                                                    circshift(envm.boundaries,-1),
                                                    circshift(envm.corners,-1),
                                                    circshift(envm.fp0,-1),
                                                    circshift(envm.fp1,-1),
                                                    circshift(envm.fp2,-1))
