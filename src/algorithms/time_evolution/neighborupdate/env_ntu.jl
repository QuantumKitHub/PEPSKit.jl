include("envtools.jl")

"""
    bondenv_NN(peps::InfinitePEPS, row::Int, col::Int)

Calculate the bond environment within "NTU-NN" approximation.
```
            (-1 +0)══(-1 +1)
                ║        ║
    (+0 -1)════Q0══   ══Q1═══(+0 +2)
                ║        ║
            (+1 +0)══(+1 +1)
```
"""
function bondenv_NN(peps::InfinitePEPS, row::Int, col::Int)
end

"""
    bondenv_NNp(peps::InfinitePEPS, row::Int, col::Int)

Calculates the metric tensor within "NTU-NN+" approximation.
```
                    (-2 +0)┈┈(-2 +1)
                        ║        ║
            (-1 -1)┈(-1 +0)══(-1 +1)┈(-1 +2)
                ┊       ║        ║       ┊
    (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)=(+0 +3)
                ┊       ║        ║       ┊
            (+1 -1)┈(+1 +0)══(+1 +1)┈(+1 +2)
                        ║        ║
                    (+2 +0)┈┈(+2 +1)
```
"""
function bondenv_NNp(peps::InfinitePEPS, row::Int, col::Int)
    return error("Not implemented")
end

"""
    bondenv_NNN(peps::InfinitePEPS, row::Int, col::Int)

Calculates the bond environment within "NTU-NNN" approximation.

```
    (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
        ║       ║        ║       ║
    (+0 -1)════Q0══   ══Q1═══(+0 +2)
        ║       ║        ║       ║
    (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)
```
"""
function bondenv_NNN(peps::InfinitePEPS, row::Int, col::Int)
    return error("Not implemented")
end

"""
    bondenv_NNNp(peps::InfinitePEPS, row::Int, col::Int)

Calculates the bond environment within "NTU-NNN+" approximation.
```
            (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                ║       ║        ║       ║
    (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)═(-1 +3)
                ║       ║        ║       ║
    (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)═(+0 +3)
                ║       ║        ║       ║
    (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                ║       ║        ║       ║
            (+2 -1) (+2 +0)  (+2 +1) (+2 +2)
```
"""
function bondenv_NNNp(peps::InfinitePEPS, row::Int, col::Int)
    return error("Not implemented")
end
