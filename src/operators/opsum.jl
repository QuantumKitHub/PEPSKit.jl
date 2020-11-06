struct OpSum{T<:Tuple} <: Operator
    ops::T
end

Base.:+(a::Operator,b::Operator) = OpSum((a,b));
Base.:+(a::Operator,b::OpSum) = OpSum((b.ops...,a))
Base.:+(b::OpSum,A::Operator) = OpSum((b.ops...,a))

Base.rotl90(st::OpSum) = st
Base.rotr90(st::OpSum) = st
