function nonsym_nn_ising_ham(;J=1.0,spin=1//2,lambda = 0.5)
    (sx,sy,sz,id) = nonsym_spintensors(spin);
    @tensor nn[-1 -2;-3 -4]:=(J*sz)[-1,-2]*sz[-3,-4]+(0.5*lambda*id)[-1,-2]*sx[-3,-4]+(0.5*lambda*sx)[-1,-2]*id[-3,-4]
    return nn
end
