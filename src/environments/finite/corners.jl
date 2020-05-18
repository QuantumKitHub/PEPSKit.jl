struct FinCorners{P,A,T}
    prevpeps :: Array{FinPeps{P},2}
    data :: Array{Array{A,1},2}

    dir :: Dir

    trscheme :: T
end

function FinCorners(peps :: FinPeps,dir::Dir,trscheme)
    tpeps = rotate_north(peps,dir);
    (tnr,tnc) = size(tpeps);

    utilleg=oneunit(space(peps[1,1],1))

    northgen = [permuteind(TensorMap(I,ComplexF64,utilleg*space(tpeps[1,j],4)',space(tpeps[1,j],4)'*utilleg),(1,2,3),(4,)) for j in 1:tnc]
    westgen = [permuteind(TensorMap(I,ComplexF64,utilleg*space(tpeps[j,1],1)',space(tpeps[j,1],1)'*utilleg),(1,2,3),(4,)) for j in 1:tnr]

    data = Array{typeof(northgen),2}(undef,tnr+1,tnc+1);
    #do le boundaries enfin
    for i in 1:tnr
        data[i+1,1]= reverse(westgen[1:i])
    end

    for j in 1:tnc
        data[1,j+1]=northgen[1:j]
    end
    data[1,1]=[];

    for i in 1:tnr
        for j in 1:tnc
            northgen = [permuteind(TensorMap(I,ComplexF64,utilleg*space(tpeps[i,tj],2),space(tpeps[i,tj],2)*utilleg),(1,2,3),(4,)) for tj in 1:j]
            westgen = [permuteind(TensorMap(I,ComplexF64,utilleg*space(tpeps[ti,j],3),space(tpeps[ti,j],3)*utilleg),(1,2,3),(4,)) for ti in i:-1:1]

            data[i+1,j+1]= [northgen;westgen]

            if length(data[i+1,j+1])>0
                data[i+1,j+1]=truncatebonds(FiniteMps(data[i+1,j+1]),trscheme).data
            end
        end
    end

    return FinCorners(map(x->similar.(x),fill(peps,tnr+1,tnc+1)),convert(Array{typeof(data[2,2]),2},data),dir,trscheme)
end

function check_corner_dependencies(peps,dir,row,col)
    (trow,tcol) = rotate_north((row,col),size(peps),dir)

    tor = []
    for i in 1:(trow)
        for j in 1:(tcol)
            push!(tor,peps[inv_rotate_north((i,j),size(peps),dir)...])
        end
    end

    return tor;
end

#(row,col) is exactly on the corner
function getdata!(env::FinCorners,row,col,peps)
    (trow,tcol) = rotate_north((row,col),size(peps),env.dir)
    oldpepsd = check_corner_dependencies(env.prevpeps[trow+1,tcol+1],env.dir,row,col)
    newpepsd = check_corner_dependencies(peps,env.dir,row,col)

    #if the data is still correct; return it
    if reduce((a,b)-> a && (b[1]===b[2]),zip(oldpepsd,newpepsd),init=true)
        return env.data[trow+1,tcol+1]
    end

    pepst= rotate_north(peps,env.dir)

    (prow,pcol) = inv_rotate_north((trow-1,tcol-1),size(peps),env.dir)
    pardat = getdata!(env::FinCorners,prow,pcol,peps);

    #gather the relevant peps tensors
    pepsl = pepst[trow,1:(tcol-1)]
    pepsc = pepst[trow,tcol]
    pepsr = pepst[(trow-1):-1:1,tcol]

    #call vomps
    newmps = vomps(pardat,pepsl#=horizonal=#,pepsc#=special=#,pepsr#=vertical=#,env.data[trow+1,tcol+1],env.trscheme);

    #update prevpeps and data
    env.data[trow+1,tcol+1] = newmps
    for i in 1:(trow)
        for j in 1:(tcol)
            env.prevpeps[trow+1,tcol+1][inv_rotate_north((i,j),size(peps),env.dir)...]=peps[inv_rotate_north((i,j),size(peps),env.dir)...]
        end
    end

    #return
    return newmps
end
