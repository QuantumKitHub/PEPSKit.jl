struct CTMRGEnv{C,T}
    #peps::InfinitePEPS{P}
    corners::Array{C,3}
    edges::Array{T,3}
end

# initialize ctmrg environments with some random tensors
function CTMRGEnv(peps::InfinitePEPS{P}) where P
    ou = oneunit(space(peps,1,1)); # the bogus space

    C_type = tensormaptype(spacetype(P),1,1,storagetype(P));
    T_type = tensormaptype(spacetype(P),3,1,storagetype(P)); # debatable how we should do the legs?

    #first index is de
    corners = Array{C_type}(undef,4,size(peps)...);
    edges = Array{T_type}(undef,4,size(peps)...);

    for dir in 1:4, i in 1:size(peps,1),j in 1:size(peps,2)
        @diffset corners[dir,i,j] = TensorMap(randn,eltype(P),ou,ou)
        @diffset edges[dir,i,j] = TensorMap(randn,eltype(P),ou*space(peps[i,j],dir+1)'*space(peps[i,j],dir+1),ou)
    end

    @diffset corners[:,:,:]./=norm.(corners[:,:,:]);
    @diffset edges[:,:,:]./=norm.(edges[:,:,:]);

    CTMRGEnv(corners,edges)
end

function Base.rotl90(envs::CTMRGEnv{C,T}) where {C,T}
    #n_peps = rotl90(envs.peps);
    n_corners = Array{C,3}(undef,size(envs.corners)...);
    n_edges = Array{T,3}(undef,size(envs.edges)...);

    for dir in 1:4
        dirm = mod1(dir-1,4)
        @diffset n_corners[dirm,:,:] .= rotl90(envs.corners[dir,:,:]);
        @diffset n_edges[dirm,:,:] .= rotl90(envs.edges[dir,:,:]);
    end

    CTMRGEnv(n_corners,n_edges)
end

Base.eltype(envs::CTMRGEnv) = eltype(envs.corners[1])
