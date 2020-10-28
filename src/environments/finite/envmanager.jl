#=
We just calculate the planes, corners and fixpoints
We assume global updates will be done (no sweeping) so we will have to recalculate everything anyway
=#

mutable struct FinEnvManager{P<:PEPSType,O<:MPSBondTensor,M<:GenericMPSTensor} <: Cache
    peps :: FinPEPS{P}

    algorithm :: MPSKit.Algorithm

    boundaries :: PeriodicArray{Vector{FiniteMPS{M,O}},1}
    corners :: PeriodicArray{Matrix{O},1}

    fp0 :: PeriodicArray{Matrix{O},1}
    fp1 :: PeriodicArray{Matrix{M},1}
end

function MPSKit.environments(peps::FinPEPS,alg::MPSKit.Algorithm)
    utilleg = oneunit(space(peps,1,1))

    # generate bogus data - maybe split this off into (planes/corners/...).jl?
    boundaries = dirmap() do dir
        tpeps = rotate_north(peps,dir)

        #this boundary vector is correct
        init = FiniteMPS([MPSKit._permute_front(isomorphism(Matrix{ComplexF64},utilleg*space(tpeps[1,j],North)',space(tpeps[1,j],North)'*utilleg)) for j in 1:size(tpeps,2)])
        dat = [init]

        #these onese are simply random
        for i in 1:size(tpeps,1)
            push!(dat,FiniteMPS([TensorMap(rand,ComplexF64,utilleg*space(tpeps[i,j],South)*space(tpeps[i,j],South)',utilleg) for j in 1:size(tpeps,2)]));
        end

        dat
    end

    corners = dirmap() do dir
        (tnr,tnc) = rotate_north(size(peps),dir)

        map(Iterators.product(1:tnr+1,1:tnc+1)) do (i,j)
            TensorMap(ones,ComplexF64,  virtualspace(boundaries[dir][i],j-1),
                                        virtualspace(boundaries[left(dir)][j],tnr-i+1))
        end
    end

    #we have to fix corners ...
    for dir in Dirs
        (tnr,tnc) = rotate_north(size(peps),dir)

        total = 1;
        for i in 1:tnr
            (ni,nj) = inv_rotate_north((i,1),size(peps),dir)
            total *= dim(space(peps[ni,nj],left(dir)))
            corners[dir][i+1,1]*=sqrt(total);
        end

        total = 1;
        for i in 1:tnc
            (ni,nj) = inv_rotate_north((1,i),size(peps),dir)
            total *= dim(space(peps[ni,nj],dir))
            corners[dir][1,i+1]*=sqrt(total);
        end
    end

    fp0 = dirmap() do dir
        (tnr,tnc) = rotate_north(size(peps),dir)

        map(Iterators.product(1:tnr+1,1:tnc+1)) do (i,j)
            TensorMap(ones,ComplexF64,  virtualspace(boundaries[left(dir)][j],tnr-i+1),
                                        virtualspace(boundaries[right(dir)][end-j+1],i-1))
        end
    end

    fp1 = dirmap() do dir
        tpeps = rotate_north(peps,dir);
        (tnr,tnc) = size(tpeps)

        map(Iterators.product(1:tnr+1,1:tnc)) do (i,j)
            psp = i == 1 ? space(tpeps[i,j],North)' : space(tpeps[i-1,j],South);
            permute(isomorphism(Matrix{ComplexF64},  virtualspace(boundaries[left(dir)][j],tnr-i+1)*psp,psp*
                                        virtualspace(boundaries[right(dir)][end-j],i-1)),(1,2,3),(4,))
        end
    end

    pars = FinEnvManager(peps,alg,boundaries,corners,fp0,fp1)

    recalculate!(pars,peps)
    pars
end

function MPSKit.recalculate!(pars::FinEnvManager,peps::FinPEPS)
    pars.peps = peps;

    plane_pars = recalc_planes!(peps,pars)
    recalc_corners!(peps,pars,plane_pars);

    recalc_fp0!(peps,pars);
    recalc_fp1!(peps,pars);
end

Base.rotl90(envm::FinEnvManager) = FinEnvManager(   rotl90(envm.peps),
                                                    envm.algorithm,
                                                    circshift(envm.boundaries,1),
                                                    circshift(envm.corners,1),
                                                    circshift(envm.fp0,1),
                                                    circshift(envm.fp1,1))
Base.rotr90(envm::FinEnvManager) = FinEnvManager(   rotr90(envm.peps),
                                                    envm.algorithm,
                                                    circshift(envm.boundaries,-1),
                                                    circshift(envm.corners,-1),
                                                    circshift(envm.fp0,-1),
                                                    circshift(envm.fp1,-1))
