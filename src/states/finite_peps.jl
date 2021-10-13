struct FinPEPS{T<:PEPSType}
    data::Matrix{T}
end

FinPEPS{T}(initializer,width,height) where T<:PEPSType = FinPEPS{T}(Array{T}(initializer,width,height))
FinPEPS(data::AbstractArray{T,2}) where T<:PEPSType = FinPEPS{T}(Array(data))

Base.copy(f::FinPEPS) = FinPEPS(copy(f.data))
Base.deepcopy(f::FinPEPS) = FinPEPS(copy.(f.data))

Base.getindex(st::FinPEPS,row,col) = st.data[row,col]
function Base.setindex!(st::FinPEPS,v,row,col)
    st.data[row,col] = v
    return st
end

Base.size(t::FinPEPS) = size(t.data)
Base.size(t::FinPEPS,i) = size(t.data,i)

Base.rotl90(dat::FinPEPS) = FinPEPS(rotl90(rotl90.(dat.data)))
Base.rotr90(dat::FinPEPS) = FinPEPS(rotr90(rotr90.(dat.data)))

#the physical space
TensorKit.space(t::FinPEPS,i,j) = space(t[i,j],5)

Base.iterate(t::FinPEPS) = iterate(t.data)
Base.iterate(t::FinPEPS,state) = iterate(t.data,state)
Base.length(t::FinPEPS) = length(t.data)
Base.map(f,x::FinPEPS) = map(f,x.data);

Base.lastindex(t::FinPEPS, i::Int64) = size(t,i)
Base.similar(t::FinPEPS) = FinPEPS(similar(t.data))
