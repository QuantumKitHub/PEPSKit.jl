struct FinPEPS{T<:PEPSType}
    data::Array{T,2}
end

Base.rotl90(dat::FinPEPS) = FinPEPS(rotl90(rotl90.(dat.data)))
Base.rotr90(dat::FinPEPS) = FinPEPS(rotl90(rotl90.(dat.data)))
