function effectivehn(channels::InfNNHamChannels,i,j)
    man = channels.envm;

    #I guess we make neff first, and init heff on zero(neff)
    utleg = isomorphism(space(man.peps[i,j],5),space(man.peps[i,j],5))

    @tensor neff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10] := man.fp1LR[West,i,j][1,-6,-1,6]*
        man.corner[SouthWest,i,j][3,1]*
        man.fp1LR[South,i,j][7,-7,-2,3]*
        man.corner[SouthEast,i,j][4,7]*
        man.fp1LR[East,i,j][8,-8,-3,4]*
        man.corner[NorthEast,i,j][5,8]*
        man.fp1LR[North,i,j][2,-9,-4,5]*
        man.corner[NorthWest,i,j][6,2]*
        utleg[-5,-10]


    heff = zero(neff)


    for dir in Dirs
        tman = rotate_north(man,dir)
        nn = rotate_north(channels.opperator,dir)
        (ti,tj) = rotate_north((i,j),size(man.peps),dir)

        tchannel = channels.ts[dir][ti,tj] #this thing is rl gauged

        #do them tchan contractions
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]:=tchannel[1,-9,-4,2]*
            tman.AR[East,ti,tj][2,-8,-3,3]*
            tman.fp1LR[South,ti,tj][3,-7,-2,4]*
            tman.AL[West,ti,tj][4,-6,-1,1]*
            utleg[-5,-10]

        #do them ham on coords - contractions
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=tman.fp1RL[North,ti-1,tj][1,2,3,4]*
            tman.AR[East,ti-1,tj][4,5,6,7]*
            tman.AR[East,ti,tj][7,-8,-3,8]*
            tman.fp1LR[South,ti,tj][8,-7,-2,9]*
            tman.AL[West,ti,tj][9,-6,-1,10]*
            tman.peps[ti-1,tj][11,-9,5,2,13]*
            tman.AL[West,ti-1,tj][10,11,12,1]*
            conj(tman.peps[ti-1,tj][12,-4,6,3,14])*
            nn[13,14,-10,-5]

        #this made sense for some reason?
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=tman.fp1RL[North,ti,tj][1,-9,-4,4]*
            tman.AR[East,ti,tj][4,-8,-3,5]*
            tman.fp1LR[South,ti,tj][5,-7,-2,6]*
            tman.AL[West,ti,tj][6,-6,-1,1]*
            utleg[-5,-10]


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
                nn[10,11,-10,-5]
        end


        #this made sense for some reason?
        @tensor cheff[-1 -2 -3 -4 -5;-6 -7 -8 -9 -10]+=fp1RL(tman,North,ti,tj)[4,-9,-4,1]*
            AR(tman,East,ti,tj)[1,-8,-3,3]*
            fp1LR(tman,South,ti,tj)[3,-7,-2,2]*
            AL(tman,West,ti,tj)[2,-6,-1,4]*
            utleg[-5,-10]

        heff +=inv_rotate_north(cheff,dir);

    end

    return (heff,neff)

end
