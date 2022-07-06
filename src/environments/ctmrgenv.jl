struct CTMRGEnv{P,C,T}
    peps::InfinitePEPS{P}
    corners::PeriodicArray{C,3}
    edges::PeriodicArray{T,3}
end


# initialize ctmrg environments with some random tensors
function CTMRGEnv(peps::InfinitePEPS{P}) where P
    ou = oneunit(space(peps,1,1)); # the bogus space

    C_type = tensormaptype(spacetype(P),1,1,storagetype(P));
    T_type = tensormaptype(spacetype(P),3,1,storagetype(P)); # debatable how we should do the legs?

    #first index is de
    corners = PeriodicArray{C_type}(undef,4,size(peps)...);
    edges = PeriodicArray{T_type}(undef,4,size(peps)...);

    for dir in 1:4, i in 1:size(peps,1),j in 1:size(peps,2)
        corners[dir,i,j] = TensorMap(randn,eltype(P),ou,ou)
        edges[dir,i,j] = TensorMap(randn,eltype(P),ou*space(peps[i,j],dir+1)'*space(peps[i,j],dir+1),ou)
    end

    CTMRGEnv(peps,corners,edges)
end

function Base.rotl90(envs::CTMRGEnv{P,C,T}) where {P,C,T}
    n_peps = rotl90(envs.peps);
    n_corners = PeriodicArray{C,3}(undef,4,size(n_peps)...);
    n_edges = PeriodicArray{T,3}(undef,4,size(n_peps)...);

    for dir in 1:4
        n_corners[dir-1,:,:] .= rotl90(envs.corners[dir,:,:]);
        n_edges[dir-1,:,:] .= rotl90(envs.edges[dir,:,:]);
    end

    CTMRGEnv(n_peps,n_corners,n_edges)
end
