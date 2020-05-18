#there are two possible hamchannels; the [1x2] and [2x1] hamchannels
#you really do want to contract them that way to keep sweeping over a single row relatively cheap

#every env has dependencies
struct FinNNHamnchannel{P,N,A}
    prevpeps1 :: Array{FinPeps{P},2}
    prevpeps2 :: Array{FinPeps{P},2}

    h1 :: Array{Tuple{Array{A,1},Array{A,1},A,Array{A,1}},2} #bit cryptic, but I can explain it on paper
    h2 :: Array{Array{A,1},2}

    #necessary corners
    tnw ::  FinCorners{P,A}
    tne ::  FinCorners{P,A}

    nn::N
end

function Base.getproperty(h::FinNNHamnchannel,f::Symbol)
    if f==:dir
        return h.tnw.dir
    else
        return getfield(h,f)
    end
end

#FinNNHamnchannel(peps :: FinPeps{P},h1 :: Array{A,2},h2 :: Array{A,2},tnw::FinCorners{P,A},tne::FinCorners{P,A}) where {P,A} = FinNNHamnchannel(peps,h1,h2,tnw,tne)
function FinNNHamnchannel(peps :: FinPeps{P},nn::NNType,tnw::FinCorners{P,A},tne::FinCorners{P,A}) where {P,A}
    @assert left(tne.dir) == tnw.dir

    #we should initialize this correctly at the edges
    (tnr,tnc) = rotate_north(size(peps),tnw.dir)
    (i,j) = inv_rotate_north((0,tnc),(tnr,tnc),tnw.dir)

    #we fill it with invalid data of the right type
    arrt = getdata!(tnw,i,j,peps)
    h1 = fill((arrt,arrt,arrt[1],arrt),tnr+1,tnc)
    h2 = fill(arrt,tnr+1,tnc)

    return FinNNHamnchannel(map(x->similar.(x),fill(peps,tnr+1,tnc)),map(x->similar.(x),fill(peps,tnr+1,tnc)),h1,h2,tnw,tne,nn)
end

function geth1!(pars::FinNNHamnchannel,row,col,peps)
    nn=pars.nn;

    (trow,tcol) = rotate_north((row,col),size(peps),pars.dir)
    #@assert trow > 1
    (nwrow,nwcol) = inv_rotate_north((trow,tcol-1),size(peps),pars.dir);
    (nerow,necol) = inv_rotate_north((trow,tcol+1),size(peps),pars.dir);

    prevdep = check_plane_dependencies(pars.prevpeps1[trow+1,tcol],pars.dir,trow+1)
    curdep = check_plane_dependencies(peps,pars.dir,trow+1)

    if reduce((a,b)-> a && (b[1]===b[2]),zip(prevdep,curdep),init=true)
        #return cached data
        #=
        if trow != 0 && tcol!=0
            tpeps = rotate_north(peps,pars.dir);
            @show space(tpeps[trow,tcol],1),space(tpeps[trow,tcol],2),space(tpeps[trow,tcol],3),space(tpeps[trow,tcol],4)
        end
        lolz = pars.h1[trow+1,tcol][4]
        @show space(lolz[1],2),space(lolz[2],2),space(lolz[3],2)
        @show trow,tcol,nwrow,nwcol,nerow,necol
        =#
        return pars.h1[trow+1,tcol]
    end

    #recalculate everything
    tpeps = rotate_north(peps,pars.dir);
    nw = getdata!(pars.tnw,nwrow,nwcol,peps);
    ne = getdata!(pars.tne,nerow,necol,peps);

    #generate the start element
    ucbt = TensorMap(LinearAlgebra.I,ComplexF64,
        space(nw[end],4)'*
        space(tpeps[1,tcol],4),
        space(tpeps[1,tcol],4)*
        space(ne[1],1));
    cbt = permuteind(ucbt,(1,3,2),(4,));
    cht = TensorMap(zeros,ComplexF64,
        space(nw[end],1)*
        space(tpeps[1,tcol],2)*
        space(tpeps[1,tcol],2)',
        space(ne[1],4)');

    #transfer to victory and beyond
    for i in 1:(trow-1)
        #transfer cht
        cht = crosstransfer(cht,[tpeps[i+1,tcol]],[ne[i+1]],[nw[end-i]])

        #hamtransfer cbt to add to cnt
        cht += hamtransfer(nw[end-i],nw[end-i+1],ne[i],ne[i+1],cbt,tpeps[i,tcol],tpeps[i+1,tcol],nn)

        #transfer cbt
        cbt = crosstransfer(cbt,[tpeps[i,tcol]],[ne[i]],[nw[end-i+1]])
    end

    pars.prevpeps1[trow+1,tcol]=copy(peps);

    pars.h1[trow+1,tcol] = (nw[1:end-trow],ne[trow+1:end],cht,[nw[end-trow+1],cbt,ne[trow]])
    return pars.h1[trow+1,tcol]
end

function geth2!(pars::FinNNHamnchannel,row,col,peps)
    nn=pars.nn
    (trow,tcol) = rotate_north((row,col),size(peps),pars.dir)
    @assert trow > 0
    (nwrow,nwcol) = inv_rotate_north((trow,tcol-1),size(peps),pars.dir);
    (nerow,necol) = inv_rotate_north((trow,tcol+2),size(peps),pars.dir);

    prevdep = check_plane_dependencies(pars.prevpeps2[trow+1,tcol],pars.dir,trow+1)
    curdep = check_plane_dependencies(peps,pars.dir,trow+1)

    if reduce((a,b)-> a && (b[1]===b[2]),zip(prevdep,curdep),init=true)
        #return cached data
        return pars.h2[trow+1,tcol]
    end

    #recalculate everything
    tpeps = rotate_north(peps,pars.dir);
    nw = getdata!(pars.tnw,nwrow,nwcol,peps);
    ne = getdata!(pars.tne,nerow,necol,peps);

    #gen start
    ucbt = TensorMap(LinearAlgebra.I,ComplexF64,
        space(nw[end],4)'*
        space(tpeps[1,tcol],4)*
        space(tpeps[1,tcol+1],4),
        space(tpeps[1,tcol],4)*
        space(tpeps[1,tcol+1],4)*
        space(ne[1],1));
    cbt = permuteind(ucbt,(1,4,2,5,3),(6,));
    cht = zero(cbt)

    for i in 1:trow
        #transfer cht
        cht=crosstransfer(cht,[tpeps[i,tcol]],[tpeps[i,tcol+1]],[ne[i]],[nw[end-i+1]])

        #hamtransfer cbt
        cht+=hamtransfer(nw[end-i+1],ne[i],cbt,tpeps[i,tcol],tpeps[i,tcol+1],nn)

        #transfer cbt
        cbt=crosstransfer(cbt,[tpeps[i,tcol]],[tpeps[i,tcol+1]],[ne[i]],[nw[end-i+1]])
    end

    #update prevpeps / h2

    #svd cht
    (U,S,V)=svd(cht,(1,2,3),(4,5,6))
    pars.h2[trow+1,tcol]=[nw[1:end-trow];U*S;permuteind(V,(1,2,3),(4,));ne[trow+1:end]]
    pars.prevpeps2[trow+1,tcol]=copy(peps);
    return pars.h2[trow+1,tcol]
end
