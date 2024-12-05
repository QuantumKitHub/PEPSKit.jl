"""
Mirror a matrix by its anti-diagonal line
(the 45 degree line through the lower-left corner)

The element originally at [r, c] is moved [Nc-c+1, Nr-r+1], 
i.e. the element now at [r, c] was originally at [Nr-c+1, Nc-r+1]
"""
function mirror_antidiag(arr::AbstractMatrix)
    Nr, Nc = size(arr)
    return collect(arr[Nr - c + 1, Nc - r + 1] for (r, c) in Iterators.product(1:Nc, 1:Nr))
end
