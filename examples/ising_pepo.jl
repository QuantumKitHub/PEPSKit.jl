using TensorOperations
using TensorKit

"""
    ising_pepo(beta; unitcell=(1, 1, 1))

Return the PEPO tensor for partition function of the 3D classical Ising model at inverse
temperature `beta`. 
"""
function ising_pepo(beta; unitcell=(1, 1, 1))
    t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
    q = sqrt(t)

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    return InfinitePEPO(O; unitcell)
end
