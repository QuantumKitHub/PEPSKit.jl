function nonsym_j1j2(j1,j2;spin=1//2)
    (sx,sy,sz,id) = nonsym_spintensors(spin);
    @tensor nn[-1 -2;-3 -4]:=sx[-1,-2]*sx[-3,-4]+sy[-1,-2]*sy[-3,-4]+sz[-1,-2]*sz[-3,-4]

    NN(j1*nn)+NNN(j2*nn)
end
