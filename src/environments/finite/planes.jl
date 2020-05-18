struct FinPlanes{P,M,T}
    prevdep :: Array{Array{P,1},1}

    data :: Array{M,2}

    dir::Dir

    trscheme :: T
end

function FinPlanes(peps::FinPeps{P},dir,trscheme) where P
    utilleg = oneunit(space(peps[1,1],1))

    tpeps = rotate_north(peps,dir)

    init = [permuteind(TensorMap(I,ComplexF64,utilleg*space(tpeps[1,j],4)',space(tpeps[1,j],4)'*utilleg),(1,2,3),(4,)) for j in 1:size(tpeps,2)]

    dat = Array{typeof(init[1]),2}(undef,size(tpeps,1)+1,size(tpeps,2))
    dat[1,:] = init;

    for i in 1:size(tpeps,1)
        dat[i+1,:] = truncatebonds(FiniteMps([TensorMap(rand,ComplexF64,utilleg*space(tpeps[i,j],2)*space(tpeps[i,j],2)',utilleg) for j in 1:size(tpeps,2)]),trscheme).data;
    end

    return FinPlanes([similar.(check_plane_dependencies(peps,dir,i)) for i in 1:size(dat,1)],dat,dir,trscheme)
end

function check_plane_dependencies(peps,dir,index)
    (tnumrows,tnumcols) = rotate_north(size(peps),dir);
    relcoords = [inv_rotate_north((a,b),size(peps),dir) for (a,b) in Iterators.product(1:index-1,1:tnumcols)]
    return map(x-> peps[x...],relcoords)[:];
end

#do the thing
function getdata!(env::FinPlanes,index,peps)
    @assert index > 0

    oldpepsd = env.prevdep[index]
    newpepsd = check_plane_dependencies(peps,env.dir,index)

    if reduce((a,b)-> a && (b[1]===b[2]),zip(oldpepsd,newpepsd),init=true)
        return env.data[index,:]
    end

    #the data is incorrect, we have to recalculate
    tpeps = rotate_north(peps,env.dir)
    (tnumrows,tnumcols) = size(tpeps)
    pdat = getdata!(env,index-1,peps)

    pepsstrip = tpeps[index-1,1:end]

    newmps = vomps(pdat,pepsstrip,env.data[index,:],env.trscheme)

    env.data[index,:] = newmps[:]

    env.prevdep[index] = check_plane_dependencies(peps,env.dir,index)

    return env.data[index,:]
end
