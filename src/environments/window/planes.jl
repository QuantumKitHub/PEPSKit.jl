function recalc_north_planes!(peps::WinPEPS,oldplanes,env_bpars,algorithm)
    #notice how oldplanes[1] is a boundary condition that cannot be changed!

    bpars = map(1:size(peps,1)) do i
        #first we create the relevant parameters
        (L,R) = env_bpars[i]
        fpars = params(oldplanes[i+1],peps[i,:],oldplanes[i],L,R)

        (oldplanes[i+1],fpars) = approximate(oldplanes[i+1],peps[i,:],oldplanes[i],algorithm,fpars)
        fpars
    end
    return bpars
end

#=
We update planes in-place, and also return the boundary environments
This will then be re-used in corners.jl
=#
function recalc_planes!(peps::WinPEPS,pars)
    tasks = map(Dirs) do dir
        ctask = @Threads.spawn begin
            tpeps = rotate_north(peps,dir);
            recalc_north_planes!(tpeps,pars.boundaries[dir],pars.infbpars[dir],pars.algorithm)
        end
    end

    bpars = fetch.(tasks)
end
