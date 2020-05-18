# In this example I go over how the InfEnvManager thing works

using TensorKit,MPSKit,PEPSKit

peps = InfPEPS(map(Iterators.product(1:1,1:1)) do (i,j)
    TensorMap(rand,ComplexF64,ℂ^3*ℂ^2*(ℂ^3)'*(ℂ^2)',ℂ^2)
end)

# we can act as if this is a 2d array
@show norm(peps[2,2]);

# with periodic boundary conditions
@show norm(peps[6,-2]);

# To practically work with this thing we use the infenvmanager object
# It can be created using
pars = params(peps);

# one sneaky caveat - it also renormalizes the peps behind the scenes
@show norm(peps[2,2]);

# we can access boundary tensors:
row = 2
col = 2
direction = North;
(AL,AC,AR) = boundary(pars,row,col,North)

# this tensor will fit on the peps tensor [row,col]
@show space(AC,2),space(peps[row,col],North)

#...
