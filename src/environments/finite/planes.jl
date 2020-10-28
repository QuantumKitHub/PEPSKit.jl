function recalc_north_planes!(peps::FinPEPS,oldplanes,algorithm)
    #notice how oldplanes[1] is a boundary condition that cannot be changed!
    pars = map(1:size(peps,1)) do i
        (oldplanes[i+1],fpars) = approximate!(oldplanes[i+1],(peps[i,:],oldplanes[i]),algorithm)
        fpars
    end
end

#=
We update planes in-place, and also return the boundary environments
This will then be re-used in corners.jl
=#
function recalc_planes!(peps::FinPEPS,pars)
    planejobs = map(Dirs) do dir
        @Threads.spawn begin
            tpeps = rotate_north(peps,dir);
            recalc_north_planes!(tpeps,pars.boundaries[dir],pars.algorithm)
        end
    end
    fetch.(planejobs)
end
