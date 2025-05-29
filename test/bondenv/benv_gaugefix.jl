using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Vphy = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1, (1, -1) => 2)
Vin = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 3, (1, -1) => 2)
V = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2, (1, -1) => 3)
Vs = (V, V')
for V1 in Vs, V2 in Vs, V3 in Vs
    #=
        ┌---┬---------------┬---┐
        |   |               |   |    ┌--------------┐
        ├---X--- -2   -3 ---Y---┤  = |              |
        |   |               |   |    └--Z-- -2  -3 -┘
        └---┴-------Z0------┴---┘       ↓
                    ↓                   -1
                    -1
    =#
    X = rand(ComplexF64, Vin ⊗ V1' ⊗ Vin' ⊗ Vin)
    Y = rand(ComplexF64, Vin ⊗ Vin ⊗ Vin' ⊗ V3)
    Z0 = randn(ComplexF64, Vphy ← Vin ⊗ Vin' ⊗ Vin ⊗ Vin ⊗ Vin ⊗ Vin')
    @tensor Z[p; Xe Yw] := Z0[p; Xn Xs Xw Yn Ye Ys] * X[Xn Xe Xs Xw] * Y[Yn Ye Ys Yw]
    #= 
        ┌---------------------------┐
        |                           |
        └---Z-- 1 --a-- 2 --b-- 3 --┘
            ↓       ↓       ↓
            -1      -2      -3
    =#
    a = randn(ComplexF64, V1 ← Vphy' ⊗ V2)
    b = randn(ComplexF64, V2 ⊗ Vphy ← V3)
    @tensor half[:] := Z[-1; 1 3] * a[1; -2 2] * b[2 -3; 3]
    Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
    @tensor half2[:] := Z2[-1; 1 3] * a2[1; -2 2] * b2[2 -3; 3]
    @test half ≈ half2
    # test gauge transformation of X, Y
    X2, Y2 = PEPSKit._fixgauge_benvXY(X, Y, Linv, Rinv)
    @tensor Z2_[p; Xe Yw] := Z0[p; Xn Xs Xw Yn Ye Ys] * X2[Xn Xe Xs Xw] * Y2[Yn Ye Ys Yw]
    @test Z2 ≈ Z2_
end
