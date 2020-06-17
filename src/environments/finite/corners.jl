#similar to the finite planes case, oldcorners[1,:] and [:,1] are boundary conditions
function recalc_northwest_corners!(peps::FinPEPS,oldcorners,nplanes,npars,wplanes,wpars)

    for (i,j) in Iterators.product(1:size(peps,1),1:size(peps,2))
        n_above = nplanes[i];
        n_below = nplanes[i+1];

        #the west density matrix comes from contracting the north boundaries
        rho_w = leftenv(npars[i],j,n_below);

        w_above = wplanes[j];
        w_below = wplanes[j+1];
        rho_n = rightenv(wpars[j],size(peps,1)-i+1,w_below)
        
        @tensor oldcorners[i+1,j+1][-1;-2] := rho_w[11,9,10,1]*oldcorners[i,j][1,8]*rho_n[8,6,4,2]*peps[i,j][9,12,5,6,7]*conj(peps[i,j][10,13,3,4,7])*
            conj(n_below.AL[j][11,12,13,-1])*conj(w_below.AR[end-i+1][-2,5,3,2])
    end
end

function recalc_corners!(peps::FinPEPS,corners,planes,plane_pars)
    for dir in Dirs
        tpeps = rotate_north(peps,dir);
        recalc_northwest_corners!(tpeps,corners[dir],planes[dir],plane_pars[dir],planes[left(dir)],plane_pars[left(dir)])
    end
end
