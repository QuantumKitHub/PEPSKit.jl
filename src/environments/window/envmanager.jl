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
    incon = 0;
    for dir in Dirs
        tman = rotate_north(man,dir);

        for i in 1:size(man.peps,1)
            for j in 1:size(man.peps,2)+1
                fp1 = fp1LR(tman,West,i,j);
                fp0 = fp0LR(tman,West,i,j);

                nw_n = corner(tman,NorthWest,i+1,j);
                nw_o = corner(tman,NorthWest,i,j);

                @tensor pred[-1 -2 -3;-4]:=fp0[-1,1]*nw_n[1,2]*AR(tman,West,i,j)[2,-2,-3,-4]
                @tensor old[-1 -2 -3;-4]:=fp1[-1,-2,-3,1]*nw_o[1,-4];

                rmul!(nw_n,dot(pred,old)/(norm(pred)*norm(pred)))

                @tensor pred[-1 -2 -3;-4]:=fp0[-1,1]*nw_n[1,2]*AR(tman,West,i,j)[2,-2,-3,-4]
                incon = max(incon,norm(old-pred)/norm(old))
            end
        end

    end
    verbose && println("total inconsistency $(incon)")
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
