function recalc_north_planes!(peps::FinPEPS,oldplanes,algorithm)
    #notice how oldplanes[1] is a boundary condition that cannot be changed!
    for i in 1:size(peps,1)
        (oldplanes[i+1],_) = approximate(oldplanes[i+1],peps[i,:],oldplanes[i],algorithm)
    end
end

function recalc_planes!(peps::FinPEPS,planes,algorithm)
    for dir in Dirs
        tpeps = rotate_north(peps,dir);
        recalc_north_planes!(tpeps,planes[dir],algorithm)
    end
end
