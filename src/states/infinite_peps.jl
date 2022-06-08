#it's also square
struct InfPEPS{T<:PEPSType}
    data::PeriodicArray{T,2}
end

InfPEPS{T}(initializer,width,height) where T<:PEPSType = InfPEPS{T}(PeriodicArray{T}(initializer,width,height))
InfPEPS(data::AbstractArray{T,2}) where T<:PEPSType = InfPEPS{T}(PeriodicArray(data))

Base.getindex(st::InfPEPS,row,col) = st.data[row,col]
function Base.setindex!(st::InfPEPS,v,row,col)
    st.data[row,col] = v
    return st
end

Base.size(t::InfPEPS) = size(t.data)
Base.size(t::InfPEPS,i) = size(t.data,i)

Base.rotl90(dat::InfPEPS) = InfPEPS(rotl90(rotl90.(dat.data)))
Base.rotr90(dat::InfPEPS) = InfPEPS(rotr90(rotr90.(dat.data)))

#the physical space
TensorKit.space(t::InfPEPS,i,j) = space(t[i,j],5)

Base.iterate(t::InfPEPS) = iterate(t.data)
Base.iterate(t::InfPEPS,state) = iterate(t.data,state)
Base.length(t::InfPEPS) = length(t.data)
Base.map(f,x::InfPEPS) = map(f,x.data);

Base.lastindex(t::InfPEPS, i::Int64) = size(t,i)
Base.similar(t::InfPEPS) = InfPEPS(similar(t.data))

Base.copy(st::InfPEPS) = InfPEPS(copy(st.data));
Base.deepcopy(st::InfPEPS) = InfPEPS(deepcopy(st.data));
