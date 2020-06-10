#similar to the finite planes case, oldcorners[1,:] and [:,1] are boundary conditions
function recalc_northwest_corners!(peps::FinPEPS,oldcorners,nplanes,wplanes)
    for (i,j) in Iterators.product(1:size(peps,1),1:size(peps,2))
        n_above = nplanes[i];
        n_below = nplanes[i+1];

        #the west density matrix comes from contracting the north boundaries
        rho_w = permute(isomorphism(Matrix{ComplexF64},virtualspace(n_below,0)*space(peps[i,1],West)',space(peps[i,1],West)'*virtualspace(n_above,0)),(1,2,3),(4,));
        rho_w = transfer_left(rho_w,peps[i,1:j-1],n_above.AL[1:j-1],n_below.AL[1:j-1])


        w_above = wplanes[j];
        w_below = wplanes[j+1];

        rho_n = permute(isomorphism(Matrix{ComplexF64},virtualspace(w_above,size(peps,1))*space(peps[1,j],North)',space(peps[1,j],North)'*virtualspace(w_below,size(peps,1))),(1,2,3),(4,))
        rho_n = transfer_right(rho_n,reverse([rotate_north(p,West) for p in peps[1:i-1,j]]),w_above.AR[end-i+2:end],w_below.AR[end-i+2:end])

        @tensor oldcorners[i+1,j+1][-1;-2] := rho_w[1,2,3,4]*oldcorners[i,j][4,5]*rho_n[5,6,7,8]*peps[i,j][2,9,10,6,11]*conj(peps[i,j][3,12,13,7,11])*
            conj(n_below.AL[j][1,9,12,-1])*conj(w_below.AR[end-i+1][-2,10,13,8])
    end
end

function recalc_corners!(peps::FinPEPS,corners,planes)
    for dir in Dirs
        tpeps = rotate_north(peps,dir);
        recalc_northwest_corners!(tpeps,corners[dir],planes[dir],planes[left(dir)])
    end
end
