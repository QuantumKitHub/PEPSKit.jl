#---- not yet refractored
#nnpars thing
struct NNFinPars{P,N,A}
    nn::N
    envm::FinEnvManager{P,A}
    channels::Periodic{FinNNHamnchannel{P,N,A},1}
end

function params(peps::FinPeps,nn::NNType,trscheme)
    envm = EnvManager(peps,trscheme);

    udat = Array{Any,1}(undef,length(Dirs))

    for dir in Dirs
        udat[dir] = FinNNHamnchannel(peps,nn,envm.corners[dir],envm.corners[right(dir)])
    end

    channels = Periodic{typeof(udat[1]),1}(udat);

    return NNFinPars(nn,envm,channels);
end

#nnprime thing
function effectivehn(peps,man::NNFinPars,coords)
    nn = man.nn;envm = man.envm;channels = man.channels;
    (nr,nc) = size(peps);(row,col) = coords;

    #this is a bit of a helper
    put = TensorMap(I,ComplexF64,space(peps[row,col],5),space(peps[row,col],5));

    udspace = oneunit(space(peps[1,1],1))

    #this serves as the boundary vector for every mps
    mpsut = TensorMap(I,ComplexF64,udspace,udspace);
    wstart = permuteind(TensorMap(I,ComplexF64,udspace*space(peps[row,1],1)',space(peps[row,1],1)'*udspace),(1,2,3),(4,))
    sstart = permuteind(TensorMap(I,ComplexF64,udspace*space(peps[end,col],2)',space(peps[end,col],2)'*udspace),(1,2,3),(4,))
    estart = permuteind(TensorMap(I,ComplexF64,udspace*space(peps[row,end],3)',space(peps[row,end],3)'*udspace),(1,2,3),(4,))
    nstart = permuteind(TensorMap(I,ComplexF64,udspace*space(peps[1,col],4)',space(peps[1,col],4)'*udspace),(1,2,3),(4,))

    #first we make neff
    neff = begin
        above = getdata!(envm.planes[North],row,peps)
        below = getdata!(envm.planes[South],nr-row+1,peps)

        lf1 = crosstransfer(wstart,peps[row,1:col-1],above[1:col-1],below[end+1-(col-1):end],dir=West)
        rf1 = crosstransfer(estart,peps[row,end:-1:col+1],below[1:end-col],above[end+1-(end-col):end],dir=East)

        @tensor toret[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]:=lf1[4,-6,-1,1]*above[col][1,-9,-4,3]*rf1[3,-8,-3,2]*below[end-col+1][2,-7,-2,4]*put[-5,-10]
    end


    heff = zero(neff);

    #h1 somewhere above");flush(stdout)
    for i in 1:nc #position of h1
        if row == 1
            break;
        end
        (L,R,h,bult) = geth1!(channels[North],row-1,i,peps);above = [L;h;R];
        below = getdata!(envm.planes[South],nr-row+1,peps)

        wm = crosstransfer(wstart,peps[row,1:col-1],above[1:col-1],below[end+1-(col-1):end],dir=West)
        em = crosstransfer(estart,peps[row,end:-1:col+1],below[1:nc-col],above[end+1-(nc-col):end],dir=East)

        nm = above[col];sm = below[nr-col+1];

        if row > 2
            @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=wm[4,-6,-1,1]*nm[1,-9,-4,3]*em[3,-8,-3,2]*sm[2,-7,-2,4]*put[-5,-10]
        end

        if i == col && row > 1#onsite contribution

            @tensor heff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=
                wm[13,-6,-1,12]*bult[1][12,5,3,2]*bult[2][2,6,4,7]*bult[3][7,8,9,14]*em[14,-8,-3,1]*sm[1,-7,-2,13]*
                peps[row-1,col][5,-9,8,6,10]*conj(peps[row-1,col][3,-4,9,4,11])*
                nn[10,11,-10,-5]

        end
    end

    #h1 somewhere below");flush(stdout)
    for i in 1:nc #position of h1
        if row == nr
            break;
        end
        above = getdata!(envm.planes[North],row,peps)
        (L,R,h,bult) = geth1!(channels[South],row+1,i,peps);below = [L;h;R];

        wm = crosstransfer(wstart,peps[row,1:col-1],above[1:col-1],below[end+1-(col-1):end],dir=West)
        em = crosstransfer(estart,peps[row,end:-1:col+1],below[1:nc-col],above[end+1-(nc-col):end],dir=East)

        nm = above[col];sm = below[nr-col+1];

        if (nr-row) >= 2

            @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=wm[4,-6,-1,1]*nm[1,-9,-4,3]*em[3,-8,-3,2]*sm[2,-7,-2,4]*put[-5,-10]
        end

        if i == col && row<nr#onsite contribution
            #=
            @show typeof(bult)
            @show row,col
            @show space(bult[1],2),space(peps[row+1,col],3)
            @show space(bult[2],2),space(peps[row+1,col],2)
            @show space(bult[3],2),space(peps[row+1,col],1)
            =#
            @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=
                wm[14,-6,-1,1]*nm[1,-9,-4,13]*em[13,-8,-3,12]*bult[1][12,7,8,9]*bult[2][9,6,4,2]*bult[3][2,5,3,14]*
                peps[row+1,col][5,6,7,-7,10]*conj(peps[row+1,col][3,4,8,-2,11])*
                nn[10,11,-10,-5]
        end
    end

    #h2 somewhere above");flush(stdout)
    for i in 1:(nc-1) #position of h2
        if !(row>1)
            break;
        end

        above = geth2!(channels[North],row-1,i,peps)
        below = getdata!(envm.planes[South],nr-row+1,peps)

        sm = crosstransfer(wstart,peps[row,1:col-1],above[1:col-1],below[end+1-(col-1):end],dir=West)
        nm = crosstransfer(estart,peps[row,end:-1:col+1],below[1:nc-col],above[end+1-(nc-col):end],dir=East)

        wm = above[col];em = below[nr-col+1];

        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=sm[4,-6,-1,1]*wm[1,-9,-4,3]*nm[3,-8,-3,2]*em[2,-7,-2,4]*put[-5,-10]
    end

    #h2 somewhere below");flush(stdout)
    for i in 2:nc #position of h2
        if !((nr-row)>0)
            break;
        end

        above = getdata!(envm.planes[North],row,peps)
        below = geth2!(channels[South],row+1,i,peps)

        sm = crosstransfer(wstart,peps[row,1:col-1],above[1:col-1],below[end+1-(col-1):end],dir=West)
        nm = crosstransfer(estart,peps[row,end:-1:col+1],below[1:nc-col],above[end+1-(nc-col):end],dir=East)

        wm = above[col];em = below[nr-col+1];

        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=sm[4,-6,-1,1]*wm[1,-9,-4,3]*nm[3,-8,-3,2]*em[2,-7,-2,4]*put[-5,-10]
    end

    #h2 to the left");flush(stdout)
    for i in (row+1,row)
        if i==1 || i==nr+1 || col==1
            continue
        end

        left = geth2!(channels[West],i,col-1,peps)
        (L,R,_,bult) = geth1!(channels[East],row,col,peps)

        s = crosstransfer(mpsut,left[1:length(R)],R)
        n = crosstransfer(mpsut,L,left[end-length(L)+1:end])

        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=left[end-row+1][1,-6,-1,2]*n[2,3]*bult[1][3,-9,-4,6]*bult[2][6,-8,-3,9]*bult[3][9,-7,-2,12]*s[12,1]*put[-5,-10]
    end


    #h2 to the right");flush(stdout)
    for i in (row,row-1)
        if i == nr || i == 0 || col == nc
            continue
        end

        right = geth2!(channels[East],i,col+1,peps)
        (L,R,_,bult) = geth1!(channels[West],row,col,peps)

        n = crosstransfer(mpsut,right[1:length(R)],R)
        s = crosstransfer(mpsut,L,right[end-length(L)+1:end])

        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=bult[2][5,-6,-1,2]*bult[3][2,-9,-4,1]*n[1,6]*right[row][6,-8,-3,3]*s[3,4]*bult[1][4,-7,-2,5]*put[-5,-10]
    end

    #h1 to the left");flush(stdout)
    for i in (row,)
        if col==1
            continue;
        end

        (lL,lR,lh,lbult) = geth1!(channels[West],row,col-1,peps)
        (rL,rR,rh,rbult) = geth1!(channels[East],row,col,peps)

        n = crosstransfer(mpsut,rL,lR);
        s = crosstransfer(mpsut,lL,rR);

        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=lh[1,-6,-1,2]*n[2,3]*rbult[1][3,-9,-4,4]*rbult[2][4,-8,-3,5]*rbult[3][5,-7,-2,6]*s[6,1]*put[-5,-10]

        #onsite contribution
        @tensor heff[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10]+=lbult[1][1,2,3,4]*lbult[2][4,5,6,7]*lbult[3][7,8,9,10]*n[10,11]*
            rbult[1][11,-9,-4,12]*rbult[2][12,-8,-3,13]*rbult[3][13,-7,-2,14]*s[14,1]*
            peps[row,col-1][5,2,-6,8,15]*conj(peps[row,col-1][6,3,-1,9,16])*
            nn[15,16,-10,-5]
    end

    #h1 to the right");flush(stdout)
    for i in (row,)
        if col == nc
            continue
        end
        (lL,lR,lh,lbult) = geth1!(channels[West],row,col,peps)
        (rL,rR,rh,rbult) = geth1!(channels[East],row,col+1,peps)

        n = crosstransfer(mpsut,rL,lR);
        s = crosstransfer(mpsut,lL,rR);

        @tensor heff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=lbult[1][1,-7,-2,2]*lbult[2][2,-6,-1,3]*lbult[3][3,-9,-4,4]*n[4,5]*rh[5,-8,-3,6]*s[6,1]*put[-5,-10]


        #onsite contribution
        @tensor heff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=lbult[1][3,-7,-2,15]*lbult[2][15,-6,-1,2]*lbult[3][2,-9,-4,1]*n[1,16]*
            rbult[1][16,4,8,9]*rbult[2][9,11,7,5]*rbult[3][5,10,6,14]*s[14,3]*
            peps[row,col+1][-8,10,11,4,12]*conj(peps[row,col+1][-3,6,7,8,13])*
            nn[12,13,-10,-5]

    end

    return (heff,neff)
end


function endens(man::NNFinPars)
    peps = man.peps;nn = man.nn;envm = man.envm;channels = man.channels;
    (nr,nc) = size(peps);

    udspace = oneunit(space(peps[1,1],1))

    #this serves as the boundary vector for every mps
    mpsut = TensorMap(I,ComplexF64,udspace,udspace);

    toret = 0.0+0im;

    for dir in [North,East]
        (tnr,tnc) = rotate_north(size(peps),dir)
        tpeps = rotate_north(peps,dir);

        for tr in 1:tnr
            for tc = 2:tnc

                (a,b) = inv_rotate_north((tr-1,tc),size(peps),dir)
                nw = getdata!(envm.corners[dir],a,b,peps)
                (a,b) = inv_rotate_north((tr,tc+1),size(peps),dir)
                ne = getdata!(envm.corners[right(dir)],a,b,peps)
                (a,b) = inv_rotate_north((tr+1,tc-1),size(peps),dir)
                se = getdata!(envm.corners[right(right(dir))],a,b,peps)
                (a,b) = inv_rotate_north((tr,tc-2),size(peps),dir)
                sw = getdata!(envm.corners[left(dir)],a,b,peps)

                n = crosstransfer(mpsut,ne[1:tr-1],nw[end+1-(tr-1):end])
                e = crosstransfer(mpsut,se[1:tnc-tc],ne[end+1-(tnc-tc):end])
                s = crosstransfer(mpsut,sw[1:tnr-tr],se[end+1-(tnr-tr):end])
                w = crosstransfer(mpsut,nw[1:tc-2],sw[end+1-(tc-2):end])

                nm1 = nw[tc-1]; nm2 = nw[tc]; em = ne[tr];
                sm2 = se[tnc-tc+2]; sm1 = se[tnc-tc+1]; wm = sw[tnr-tr+1];


                locen = @tensor nm1[26,2,3,19]*nm2[19,10,5,27]*n[27,28]*em[28,9,12,6]*e[6,7]*sm1[7,8,11,20]*sm2[20,21,22,23]*s[23,24]*wm[24,4,1,25]*w[25,26]*
                    tpeps[tr,tc-1][4,21,17,2,18]*conj(tpeps[tr,tc-1][1,22,15,3,16])*nn[18,16,13,14]*
                    tpeps[tr,tc][17,8,9,10,13]*conj(tpeps[tr,tc][15,11,12,5,14])


                locno = @tensor nm1[26,2,3,19]*nm2[19,10,5,27]*n[27,28]*em[28,9,12,6]*e[6,7]*sm1[7,8,11,20]*sm2[20,21,22,23]*s[23,24]*wm[24,4,1,25]*w[25,26]*
                    tpeps[tr,tc-1][4,21,17,2,16]*conj(tpeps[tr,tc-1][1,22,15,3,16])*
                    tpeps[tr,tc][17,8,9,10,13]*conj(tpeps[tr,tc][15,11,12,5,13])

                toret += locen/locno
            end
        end
    end

    toret
end
