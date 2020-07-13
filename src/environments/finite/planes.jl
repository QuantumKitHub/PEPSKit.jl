function recalc_north_planes!(peps::FinPEPS,oldplanes,algorithm)
    #notice how oldplanes[1] is a boundary condition that cannot be changed!

    bpars = map(1:size(peps,1)) do i
        (oldplanes[i+1],fpars) = approximate(oldplanes[i+1],(peps[i,:],oldplanes[i]),algorithm)
        fpars[1]
    end
    return bpars
end

#=
We update planes in-place, and also return the boundary environments
This will then be re-used in corners.jl
=#
function recalc_planes!(peps::FinPEPS,pars)
    tasks = map(Dirs) do dir
        ctask = @Threads.spawn begin
            tpeps = rotate_north(peps,dir);
            recalc_north_planes!(tpeps,pars.boundaries[dir],pars.algorithm)
        end
    end

    bpars = fetch.(tasks)
end
