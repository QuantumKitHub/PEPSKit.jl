module BenchUtils

export tomlify, untomlify

using TensorKit
using PEPSKit
using SUNRepresentations  # provides SU3Irrep / SU₃ for parsed space strings
using TOML

tomlify(V::VectorSpace) = sprint(show, V; context = :limited => false)
untomlify(::Type{<:VectorSpace}, s::AbstractString) = eval(Meta.parse(s))

end
