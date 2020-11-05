struct NNN{T<:AbstractTensorMap} <: Operator
    o::T
end

Base.rotl90(st::NNN) = st
Base.rotr90(st::NNN) = st
