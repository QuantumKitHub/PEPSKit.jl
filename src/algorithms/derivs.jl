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
            nn.o[11,10,-5,-10]

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
                nn.o[11,10,-5,-10]
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
                nn.o[11,10,-5,-10]
        end

        heff +=inv_rotate_north(cheff,dir);
    end
    return (heff,neff)

end

function effectivehn(channels::InfNNNHamChannels,i,j)
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
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] += fp1LR(tman,West,ti,tj)[2,-6,-1,3]*
            AL(tman,North,ti,tj)[3,-9,-4,22]*
            AC(tman,North,ti,tj+1)[22,9,7,5]*
            fp1LR(tman,East,ti,tj+1)[5,8,6,4]*
            corner(tman,SouthEast,ti,tj+1)[4,21]*
            AR(tman,East,ti+1,tj+1)[21,13,18,11]*
            fp1LR(tman,South,ti+1,tj+1)[11,12,17,14]*
            AL(tman,West,ti+1,tj+1)[14,15,16,23]*
            corner(tman,SouthWest,ti,tj+1)[23,1]*
            AR(tman,South,ti,tj)[1,-7,-2,2]*
            tman.peps[ti,tj+1][-8,19,8,9,10]*
            conj(tman.peps[ti,tj+1][-3,20,6,7,10])*
            tman.peps[ti+1,tj+1][15,12,13,19,25]*
            conj(tman.peps[ti+1,tj+1][16,17,18,20,24])*
            nn.o[-5,-10,24,25]


        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] +=
            fp1LR(tman,West,ti,tj-1)[12,17,13,15]*
            AL(tman,North,ti,tj-1)[15,19,16,25]*
            AC(tman,North,ti,tj)[25,-9,-4,2]*
            fp1LR(tman,East,ti,tj)[2,-8,-3,1]*
            corner(tman,SouthEast,ti,tj)[1,26]*
            AR(tman,East,ti+1,tj)[26,5,10,3]*
            fp1LR(tman,South,ti+1,tj)[3,4,9,6]*
            AL(tman,West,ti+1,tj)[6,7,8,23]*
            corner(tman,SouthWest,ti,tj)[23,11]*
            AR(tman,South,ti,tj-1)[11,18,14,12]*
            tman.peps[ti,tj-1][17,18,-6,19,21]*
            conj(tman.peps[ti,tj-1][13,14,-1,16,20])*
            tman.peps[ti+1,tj][7,4,5,-7,24]*
            conj(tman.peps[ti+1,tj][8,9,10,-2,22])*
            nn.o[20,21,22,24]*
            utleg[-5 -10]

        heff +=inv_rotate_north(cheff,dir);
    end

    return (heff,neff)
end


function effectivehn(e::OpSumEnv,i,j)
    res = map(t->effectivehn(t,i,j),e.envs) # should do this in parallel
    sum(first.(res)),sum(last.(res))/length(e.envs)
end
