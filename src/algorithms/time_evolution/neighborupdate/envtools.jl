"""
    cal_envboundary(free_axs::Int, ket::PEPSTensor, bra::PEPSTensor)

Contract the physical axes and the virtual axes of a PEPSTensor `ket` with `bra` to obtain the tensor on the boundary of the bond environment. 

# Examples 

- Left "hair" tensor (`free_ax = 3`)
(o is the parity tensor to cancel unwanted fermion sign)
```
             ↗|
    |--→--bra--→-- 1
    |    ↗ |  |
    |   |  |  o
    |   |  ↑  |
    |   |  |  |
    |   |  | ↙
    |--←|-ket--←-- 2
        |↙
```

- Upper-left corner tensor (`free_ax = [3, 4]`)
```
             ↗|
    |--→--bra--→-- 1
    |    ↗ |  |
    |   2  |  o
    |      ↑  |
    |      |  |
    |      | ↙
    |--←--ket--←-- 3
         ↙
        4
```

- Left edge tensor (`free_ax = [2, 3, 4]`)
```
                1
              ↗
    |--→---bra---→-- 2
    |     ↗ |
    |   3   |
    |       ↑
    |       |   4
    |       | ↙
    |--←---ket---←-- 5
          ↙
        6
```
"""
function cal_envboundary(free_axs::Vector{Int}, ket::PEPSTensor, bra::PEPSTensor)
    @assert all(2 <= ax <= 5 for ax in free_axs)
    @assert [isdual(space(ket, ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
    @assert [isdual(space(bra, ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
    codomain_axes = Tuple(ax for ax in 1:5 if !(ax in free_axs))
    domain_axes = Tuple(free_axs)
    perm = (codomain_axes, domain_axes)
    t = adjoint(permute(bra, perm)) * permute(ket, perm)
    return t
end
