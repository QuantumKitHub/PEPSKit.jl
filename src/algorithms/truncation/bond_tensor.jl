"""
Given the first tensor `A` in the cluster acted on by a gate,
obtain reduced tensor on its next bond.

For PEPSTensor,
```
        1
        |
    4 - X ← 2   1 ← a - 3
        |            ↘
        3             2
```
For PEPOTensor,
```
    gate_ax = 1                 gate_ax = 2

     1  2                           2         2      
      ↘ |                           |          ↘     
    5 - X ← 3   1 ← a - 3       5 - X ← 3   1 ← a - 3
        |            ↘              | ↘              
        4             2             4  1             
```
"""
function bond_tensor_first(A::PEPSTensor; gate_ax::Integer = 1, kwargs...)
    @assert gate_ax == 1
    X, a = left_orth!(permute(A, ((2, 4, 5), (1, 3)); copy = true); kwargs...)
    X = permute(X, (1, 4, 2, 3))
    a = permute(a, ((1, 2), (3,)))
    return X, a
end
function bond_tensor_first(A::PEPOTensor; gate_ax::Integer = 1, kwargs...)
    @assert 1 <= gate_ax <= 2
    X, a = if gate_ax == 1
        left_orth!(permute(A, ((2, 3, 5, 6), (1, 4)); copy = true); kwargs...)
    else
        left_orth!(permute(A, ((1, 3, 5, 6), (2, 4)); copy = true); kwargs...)
    end
    X = permute(X, (1, 2, 5, 3, 4))
    a = permute(a, ((1, 2), (3,)))
    return X, a
end

"""
Undo the decomposition in `bond_tensor_first`.
"""
function undo_bond_tensor_first(X::PEPSOrth, a::MPSTensor; gate_ax::Integer = 1)
    @assert gate_ax == 1
    return @tensor A[-1; -2 -3 -4 -5] := X[-2 1 -4 -5] * a[1 -1 -3]
end
function undo_bond_tensor_first(X::PEPOOrth, a::MPSTensor; gate_ax::Integer = 1)
    @assert 1 <= gate_ax <= 2
    if gate_ax == 1
        return @tensor A[-1 -2; -3 -4 -5 -6] := X[-2 -3 1 -5 -6] * a[1 -1 -4]
    else
        return @tensor A[-1 -2; -3 -4 -5 -6] := X[-1 -3 1 -5 -6] * a[1 -2 -4]
    end
end

"""
Given the last tensor `A` in the cluster acted on by a gate,
obtain reduced tensor on its previous bond.

For PEPSTensor,
```
                    1
                    |
    1 - b → 3   4 → Y - 2
          ↘         |
            2       3
```
For PEPOTensor,
```
    gate_ax = 1                 gate_ax = 2

                 1  2             2             2
                  ↘ |              ↘            |
    1 - b → 3   5 → Y - 3       1 - b → 3   5 → Y - 3
         ↘          |                           | ↘
          2         4                           4  1
```
"""
function bond_tensor_last(A::PEPSTensor; gate_ax::Integer = 1, kwargs...)
    @assert gate_ax == 1
    Y, b = left_orth!(permute(A, ((2, 3, 4), (1, 5)); copy = true); kwargs...)
    Y = permute(Y, (1, 2, 3, 4))
    b = permute(b, ((3, 2), (1,)))
    return Y, b
end
function bond_tensor_last(A::PEPOTensor; gate_ax::Integer = 1, kwargs...)
    @assert 1 <= gate_ax <= 2
    Y, b = if gate_ax == 1
        left_orth!(permute(A, ((2, 3, 4, 5), (1, 6)); copy = true); kwargs...)
    else
        left_orth!(permute(A, ((1, 3, 4, 5), (2, 6)); copy = true); kwargs...)
    end
    Y = permute(Y, (1, 2, 3, 4, 5))
    b = permute(b, ((3, 2), (1,)))
    return Y, b
end

"""
Undo the decomposition in `bond_tensor_last`.
"""
function undo_bond_tensor_last(Y::PEPSOrth, b::MPSTensor)
    return @tensor A[-1; -2 -3 -4 -5] := b[-5 -1 1] * Y[-2 -3 -4 1]
end
function undo_bond_tensor_last(Y::PEPOOrth, b::MPSTensor; gate_ax::Integer = 1)
    @assert 1 <= gate_ax <= 2
    if gate_ax == 1
        return @tensor A[-1 -2; -3 -4 -5 -6] := b[-6 -1 1] * Y[-2 -3 -4 -5 1]
    else
        return @tensor A[-1 -2; -3 -4 -5 -6] := b[-6 -2 1] * Y[-1 -3 -4 -5 1]
    end
end
