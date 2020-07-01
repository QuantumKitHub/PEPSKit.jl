mutable struct WinEnvManager{P<:WinPEPS,A<:InfEnvManager,B,S<:MPSComoving,O<:MPSKit.MPSBondTensor,M<:MPSKit.GenericMPSTensor} <: Cache
    peps :: P

    infenvm :: A
    infbpars :: PeriodicArray{B,1} #todo; type this properly
    algorithm :: MPSKit.Algorithm

    boundaries :: PeriodicArray{Vector{S},1}
    corners :: PeriodicArray{Matrix{O},1}

    fp0 :: PeriodicArray{Matrix{O},1}
    fp1 :: PeriodicArray{Matrix{M},1}
end

MPSKit.params(peps::WinPEPS,alg::MPSKit.Algorithm;kwargs...) = params(peps,params(peps.outside;kwargs...),alg);
function MPSKit.params(peps::WinPEPS,inf_peps_args::InfEnvManager,alg::MPSKit.Algorithm)
    utilleg = oneunit(space(peps,1,1))

    # generate bogus data - maybe split this off into (planes/corners/...).jl?
    boundaries = PeriodicArray(map(Dirs) do dir
        tpeps = rotate_north(peps,dir)

        #this boundary vector is correct
        inf_bounds = PeriodicArray(convert(Vector,inf_peps_args.boundaries[dir]));
        init = MPSComoving(inf_bounds[1],size(tpeps,2));
        dat = [init]

        #these onese are simply random
        for i in 1:size(tpeps,1)
            pspaces = [space(tpeps[i,j],South)*space(tpeps[i,j],South)' for j in 1:size(tpeps,2)];
            push!(dat,MPSComoving(rand,ComplexF64,pspaces,virtualspace(inf_bounds[i+1],0),inf_bounds[i+1],inf_bounds[i+1]));
        end

        dat
    end)

    inf_bpars = PeriodicArray(map(Dirs) do dir
        (tnr,tnc) = rotate_north(size(peps),dir)
        par = params(inf_peps_args.boundaries[dir],rotate_north(peps.outside,dir))

        PeriodicArray(map(1:tnr) do i
            (leftenv(par,i,1,inf_peps_args.boundaries[dir]),rightenv(par,i,tnc,inf_peps_args.boundaries[dir]))
        end)
    end)

    corners = PeriodicArray(map(Dirs) do dir
        (tnr,tnc) = rotate_north(size(peps),dir)
        copy.(inf_peps_args.corners[dir][1:tnr+1,1:tnc+1])
    end)

    fp0 = PeriodicArray(map(Dirs) do dir
        (tnr,tnc) = rotate_north(size(peps),dir)

        copy.(inf_peps_args.fp0[dir][1:tnr+1,1:tnc+1])
    end)

    fp1 = PeriodicArray(map(Dirs) do dir
        (tnr,tnc) = rotate_north(size(peps),dir)

        copy.(inf_peps_args.fp1[dir][1:tnr+1,1:tnc])
    end)

    pars = WinEnvManager(peps,inf_peps_args,inf_bpars,alg,boundaries,corners,fp0,fp1)

    MPSKit.recalculate!(pars,peps)
    pars
end

function MPSKit.recalculate!(pars::WinEnvManager,peps::WinPEPS)
    pars.peps = peps;

    plane_pars = recalc_planes!(peps,pars);
    recalc_corners!(peps,pars,plane_pars);

    recalc_fp0!(peps,pars);
    recalc_fp1!(peps,pars);

    fixphases!(pars);
end

function fixphases!(man::WinEnvManager;verbose=false)
# todo : clean this up
    for (i,j) in Iterators.product(2:size(man.peps,1),2:size(man.peps,2))
        nw = corner(man,NorthWest,i,j);
        ne = corner(man,NorthEast,i,j-1);
        se = corner(man,SouthEast,i-1,j-1);
        sw = corner(man,SouthWest,i-1,j);

        n0 = fp0LR(man,North,i,j);
        e0 = fp0LR(man,East,i,j-1);
        s0 = fp0LR(man,South,i-1,j-1);
        w0 = fp0LR(man,West,i-1,j);

        n = CR(man,North,i,j-1);
        e = CR(man,East,i-1,j-1);
        s = CR(man,South,i-1,j);
        w = CR(man,West,i,j);

        a = dot(n,nw*n0*ne)/(norm(n)*norm(n));
        b = dot(e,ne*e0*se)/(norm(e)*norm(e));
        c = dot(s,se*s0*sw)/(norm(s)*norm(s));
        d = dot(w,sw*w0*nw)/(norm(w)*norm(w));

        #a/=abs(a);b/=abs(b);c/=abs(c);d/=abs(d);
        #an approximate solution for the problem is given by:
        rmul!(ne,1/a);
        rmul!(se,d/c)
        rmul!(sw,1/d)

        #these are all the consistency equations we impose in this step : ...
        verbose && println("fp0 - nw inconsistency $(norm(nw*n0*ne-n))")
        verbose && println("fp0 - ne inconsistency $(norm(ne*e0*se-e))")
        verbose && println("fp0 - se inconsistency $(norm(se*s0*sw-s))")
        verbose && println("fp0 - sw inconsistency $(norm(sw*w0*nw-w))")
    end

    for dir in Dirs
        tman = rotate_north(man,dir);

        i = 1;
        for j in 1:size(tman.peps,2)
            nw = corner(tman,NorthWest,i,j);
            ne = corner(tman,NorthEast,i,j-1);
            se = corner(tman,SouthEast,i-1,j-1);
            sw = corner(tman,SouthWest,i-1,j);

            n0 = fp0LR(tman,North,i,j);
            e0 = fp0LR(tman,East,i,j-1);
            s0 = fp0LR(tman,South,i-1,j-1);
            w0 = fp0LR(tman,West,i-1,j);

            n = CR(tman,North,i,j-1);
            e = CR(tman,East,i-1,j-1);
            s = CR(tman,South,i-1,j);
            w = CR(tman,West,i,j);

            a = dot(n,nw*n0*ne)/(norm(n)*norm(n));
            b = dot(e,ne*e0*se)/(norm(e)*norm(e));
            c = dot(s,se*s0*sw)/(norm(s)*norm(s));
            d = dot(w,sw*w0*nw)/(norm(w)*norm(w));


            if j == 1
                rmul!(se,1/b);
            else
                rmul!(se,1/b);
                rmul!(sw,1/d);
            end

            verbose && println("fp0 - nw inconsistency $(norm(nw*n0*ne-n))")
            verbose && println("fp0 - ne inconsistency $(norm(ne*e0*se-e))")
            verbose && println("fp0 - se inconsistency $(norm(se*s0*sw-s))")
            verbose && println("fp0 - sw inconsistency $(norm(sw*w0*nw-w))")
        end
    end
end


Base.rotl90(envm::WinEnvManager) = WinEnvManager(   rotl90(envm.peps),
                                                    rotl90(envm.infenvm),
                                                    circshift(envm.infbpars,1),
                                                    envm.algorithm,
                                                    circshift(envm.boundaries,1),
                                                    circshift(envm.corners,1),
                                                    circshift(envm.fp0,1),
                                                    circshift(envm.fp1,1))
Base.rotr90(envm::WinEnvManager) = WinEnvManager(   rotr90(envm.peps),
                                                    rotr90(envm.infenvm),
                                                    circshift(envm.infbpars,-1),
                                                    envm.algorithm,
                                                    circshift(envm.boundaries,-1),
                                                    circshift(envm.corners,-1),
                                                    circshift(envm.fp0,-1),
                                                    circshift(envm.fp1,-1))
