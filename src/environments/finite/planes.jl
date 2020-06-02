struct FinPlanes{P,M}
    prevdep :: Vector{Vector{P}} # the peps that was used to generate the boundaries
    data :: Vector{M}
    dir::Dir
    alg::MPSKit.Algorithm
end

function FinPlanes(peps::FinPEPS,dir::Dir,alg::MPSKit.Algorithm)
    utilleg = oneunit(space(peps,1,1))

    tpeps = rotate_north(peps,dir)

    #this boundary vector is correct
    init = FiniteMPS([MPSKit._permute_front(isomorphism(Matrix{ComplexF64},utilleg*space(tpeps[1,j],North)',space(tpeps[1,j],North)'*utilleg)) for j in 1:size(tpeps,2)])
    dat = [init]

    #these onese are simply random
    for i in 1:size(tpeps,1)
        push!(dat,FiniteMPS([TensorMap(rand,ComplexF64,utilleg*space(tpeps[i,j],South)*space(tpeps[i,j],South)',utilleg) for j in 1:size(tpeps,2)]));
    end

    return FinPlanes([similar.(check_plane_dependencies(peps,dir,i)) for i in 1:size(dat,1)],dat,dir,alg)
end

#coming from direction 'dir', take the index-finitemps, on what tensors does it depend?
function check_plane_dependencies(peps,dir,index)
    (tnumrows,tnumcols) = rotate_north(size(peps),dir);
    relcoords = [inv_rotate_north((a,b),size(peps),dir) for (a,b) in Iterators.product(1:index-1,1:tnumcols)]
    return map(x-> peps[x...],relcoords)[:];
end

function Base.getindex(env::FinPlanes,peps::FinPEPS,index::Int64)
    index > 0 || throw(BoundsError(FinPlanes,index))

    oldpepsd = env.prevdep[index]
    newpepsd = check_plane_dependencies(peps,env.dir,index)
    reduce((a,b)-> a && (b[1]===b[2]),zip(oldpepsd,newpepsd),init=true) || recalculate!(env,index,peps)

    return env.data[index]
end

function recalculate!(env::FinPlanes,index,peps)
    #data of the row just above this one
    pdat = env[peps,index-1]

    tpeps = rotate_north(peps,env.dir)
    (tnumrows,tnumcols) = size(tpeps)

    (env.data[index],_) = approximate(env.data[index],tpeps[index-1,:],pdat,env.alg)
    env.prevdep[index] = check_plane_dependencies(peps,env.dir,index)
end
