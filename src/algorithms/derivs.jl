function effectivehn(channels::Union{WinNNHamChannels,InfNNHamChannels},i,j)
    man = channels.envm;

    #I guess we make neff first, and init heff on zero(neff)
    utleg = isomorphism(space(man.peps[i,j],5),space(man.peps[i,j],5))

    @tensor neff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] := fp1LR(man,West,i,j)[1,-6,-1,6]*
        corner(man,SouthWest,i,j)[3,1]*
        fp1LR(man,South,i,j)[7,-7,-2,3]*
        corner(man,SouthEast,i,j)[4,7]*
        fp1LR(man,East,i,j)[8,-8,-3,4]*
        corner(man,NorthEast,i,j)[5,8]*
        fp1LR(man,North,i,j)[2,-9,-4,5]*
        corner(man,NorthWest,i,j)[6,2]*
        utleg[-5,-10]


    heff = zero(neff)


    for dir in Dirs
        tman = rotate_north(man,dir)
        nn = rotate_north(channels.opperator,dir)
        (ti,tj) = rotate_north((i,j),size(man.peps),dir)

        tchannel = channels.ts[dir][ti,tj] #this thing is rl gauged

        #do them tchan contractions
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]:=tchannel[4,-9,-4,1]*
            AR(tman,East,ti,tj)[1,-8,-3,3]*
            fp1LR(tman,South,ti,tj)[3,-7,-2,2]*
            AL(tman,West,ti,tj)[2,-6,-1,4]*
            utleg[-5,-10]

        #do them ham on coords - contractions

        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=
            fp1LR(tman,North,ti-1,tj)[9,6,4,2]*
            AL(tman,East,ti-1,tj)[2,5,3,13]*
            AC(tman,East,ti,tj)[13,-8,-3,1]*
            fp1LR(tman,South,ti,tj)[1,-7,-2,14]*
            AC(tman,West,ti,tj)[14,-6,-1,12]*
            tman.peps[ti-1,tj][7,-9,5,6,10]*
            AR(tman,West,ti-1,tj)[12,7,8,9]*
            conj(tman.peps[ti-1,tj][8,-4,3,4,11])*
            nn[11,10,-5,-10]

        heff +=inv_rotate_north(cheff,dir);
    end

    return (heff,neff)
end

function effectivehn(channels::FinNNHamChannels,i,j)
    man = channels.envm;

    #I guess we make neff first, and init heff on zero(neff)
    utleg = isomorphism(Matrix{ComplexF64},space(man.peps[i,j],5),space(man.peps[i,j],5))

    @tensor neff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] := fp1LR(man,West,i,j)[1,-6,-1,6]*
        corner(man,SouthWest,i,j)[3,1]*
        fp1LR(man,South,i,j)[7,-7,-2,3]*
        corner(man,SouthEast,i,j)[4,7]*
        fp1LR(man,East,i,j)[8,-8,-3,4]*
        corner(man,NorthEast,i,j)[5,8]*
        fp1LR(man,North,i,j)[2,-9,-4,5]*
        corner(man,NorthWest,i,j)[6,2]*
        utleg[-5,-10]

    heff = zero(neff)

    for dir in Dirs
        tman = rotate_north(man,dir)
        nn = rotate_north(channels.opperator,dir)
        (ti,tj) = rotate_north((i,j),size(man.peps),dir)

        tchannel = channels.ts[dir][ti,tj] #this thing is rl gauged


        #do them tchan contractions
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]:=tchannel[4,-9,-4,1]*
            AR(tman,East,ti,tj)[1,-8,-3,3]*
            fp1LR(tman,South,ti,tj)[3,-7,-2,2]*
            AL(tman,West,ti,tj)[2,-6,-1,4]*
            utleg[-5,-10]

        if ti > 1
            #do them ham on coords - contractions
            @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=
                fp1RL(tman,North,ti-1,tj)[9,6,4,2]*
                AR(tman,East,ti-1,tj)[2,5,3,13]*
                AR(tman,East,ti,tj)[13,-8,-3,1]*
                fp1LR(tman,South,ti,tj)[1,-7,-2,14]*
                AL(tman,West,ti,tj)[14,-6,-1,12]*
                tman.peps[ti-1,tj][7,-9,5,6,10]*
                AL(tman,West,ti-1,tj)[12,7,8,9]*
                conj(tman.peps[ti-1,tj][8,-4,3,4,11])*
                nn[11,10,-5,-10]
        end

        heff +=inv_rotate_north(cheff,dir);

    end

    return (heff,neff)

end

function effectivehn(cors::FinNNHamCors,i,j)
    man = cors.envm;

    #I guess we make neff first, and init heff on zero(neff)
    utleg = isomorphism(Matrix{ComplexF64},space(man.peps[i,j],5),space(man.peps[i,j],5))

    @tensor neff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] :=
    fp1LR(man,North,i,j)[4,-9,-4,1]*
    AC(man,East,i,j)[1,-8,-3,3]*
    fp1LR(man,South,i,j)[3,-7,-2,2]*
    AC(man,West,i,j)[2,-6,-1,4]*
    utleg[-5,-10]

    @tensor neff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] +=
    AC(man,North,i,j)[4,-9,-4,1]*
    fp1LR(man,East,i,j)[1,-8,-3,3]*
    AC(man,South,i,j)[3,-7,-2,2]*
    fp1LR(man,West,i,j)[2,-6,-1,4]*
    utleg[-5,-10]

    neff/=2;

    heff = zero(neff)
    for dir in Dirs
        tman = rotate_north(man,dir)
        nn = rotate_north(cors.opperator,dir)
        (tnr,tnc) = rotate_north(size(man.peps),dir);
        (ti,tj) = rotate_north((i,j),size(man.peps),dir)

        #do them tchan contractions
        hamline = cors.lines[dir][ti,tj]
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]:=hamline[4,-9,-4,1]*
            AC(tman,East,ti,tj)[1,-8,-3,3]*
            fp1LR(tman,South,ti,tj)[3,-7,-2,2]*
            AC(tman,West,ti,tj)[2,-6,-1,4]*
            utleg[-5,-10]

        hamr = cors.cors[right(dir)][end-tj];
        bl = man.boundaries[left(dir)][tj];
        n = fp1LR(tman,North,1,tj);
        n = crosstransfer(n,tman.peps[1:ti-1,tj],hamr.AL[1:ti-1],bl.AR[end:-1:end-ti+2])
        s = fp1LR(tman,South,tnr,tj);
        s = crosstransfer(s,tman.peps[end:-1:ti+1,tj],bl.AL[1:end-ti],hamr.AR[end:-1:ti+1],dir=South);

        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] += n[4,-9,-4,1]*hamr.AC[ti][1,-8,-3,3]*s[3,-7,-2,2]*bl.AC[end-ti+1][2,-6,-1,4]*utleg[-5,-10]

        if ti > 1
            #do them ham on coords - contractions
            @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=
                fp1RL(tman,North,ti-1,tj)[9,6,4,2]*
                AR(tman,East,ti-1,tj)[2,5,3,13]*
                AR(tman,East,ti,tj)[13,-8,-3,1]*
                fp1LR(tman,South,ti,tj)[1,-7,-2,14]*
                AL(tman,West,ti,tj)[14,-6,-1,12]*
                tman.peps[ti-1,tj][7,-9,5,6,10]*
                AL(tman,West,ti-1,tj)[12,7,8,9]*
                conj(tman.peps[ti-1,tj][8,-4,3,4,11])*
                nn[11,10,-5,-10]
        end

        heff +=inv_rotate_north(cheff,dir);
    end
    return (heff,neff)

end
