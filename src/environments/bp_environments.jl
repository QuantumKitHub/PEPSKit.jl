struct BPEnv{T}
    "4 x rows x cols array of message tensors, where the first dimension specifies the spactial direction"
    messages::Array{T,3}
end

