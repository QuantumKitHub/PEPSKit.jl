using MPSKit: InfiniteEnvironments

# overloads required purely because of the fact that left and right virtual spaces are now
# ProductSpace instances

function MPSKit.issamespace(
    env::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS,
)
    L = MPSKit.check_length(above, operator, below)
    for i in 1:L
        space(env.GLs[i]) == (
            left_virtualspace(below, i) ⊗
            _elementwise_dual(left_virtualspace(operator, i)) ← left_virtualspace(above, i)
        ) || return false
        space(env.GRs[i]) == (
            right_virtualspace(above, i) ⊗ right_virtualspace(operator, i) ←
            right_virtualspace(below, i)
        ) || return false
    end
    return true
end

function MPSKit.allocate_GL(
    bra::InfiniteMPS, mpo::InfiniteTransferMatrix, ket::InfiniteMPS, i::Int
)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V =
        left_virtualspace(bra, i) ⊗ _elementwise_dual(left_virtualspace(mpo, i)) ←
        left_virtualspace(ket, i)
    TT = TensorMap{T}
    return TT(undef, V)
end

function MPSKit.allocate_GR(
    bra::InfiniteMPS, mpo::InfiniteTransferMatrix, ket::InfiniteMPS, i::Int
)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i) ← right_virtualspace(bra, i)
    TT = TensorMap{T}
    return TT(undef, V)
end
