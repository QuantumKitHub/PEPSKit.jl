"""
Convert Hamiltonian `H` with nearest neighbor terms to `exp(-dt * H)`
"""
function get_gate(dt::Float64, H::LocalOperator)
    return LocalOperator(
        H.lattice, Tuple(sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
end

"""
Check if two 2-site bonds are related by a (periodic) lattice translation
"""
function is_equivalent(
    bond1::NTuple{2,CartesianIndex{2}},
    bond2::NTuple{2,CartesianIndex{2}},
    (Nrow, Ncol)::NTuple{2,Int},
)
    r1 = bond1[1] - bond1[2]
    r2 = bond2[1] - bond2[2]
    shift_row = bond1[1][1] - bond2[1][1]
    shift_col = bond1[1][2] - bond2[1][2]
    return r1 == r2 && mod(shift_row, Nrow) == 0 && mod(shift_col, Ncol) == 0 
end

"""
Get the term of a 2-site gate acting on a certain bond.
Input `gate` should only include one term for each nearest neighbor bond.
"""
function get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})
    label = findall(p -> is_equivalent(p.first, bond, size(gate.lattice)), gate.terms)
    if length(label) == 0
        # try reversed site order
        label = findall(
            p -> is_equivalent(p.first, reverse(bond), size(gate.lattice)), gate.terms
        )
        @assert length(label) == 1
        return permute(gate.terms[label[1]].second, ((2, 1), (4, 3)))
    else
        @assert length(label) == 1
        return gate.terms[label[1]].second
    end
end

"""
Get the position of `site` after reflection about the anti-diagonal line
"""
function _mirror_antidiag_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 - c + Ncol, 1 - r + Nrow)
end

"""
Get the position of `site` after clockwise (right) rotation by 90 degrees
"""
function _rotr90_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(c, 1 + Nrow - r)
end

"""
Get the position of `site` after counter-clockwise (left) rotation by 90 degrees
"""
function _rotl90_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 + Ncol - c, r)
end

"""
Get the position of `site` after rotation by 180 degrees
"""
function _rot180_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 + Nrow - r, 1 + Ncol - c)
end

function mirror_antidiag(H::LocalOperator)
    lattice2 = mirror_antidiag(H.lattice)
    terms2 = (
        (Tuple(_mirror_antidiag_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rotr90(H::LocalOperator)
    lattice2 = rotr90(H.lattice)
    terms2 = (
        (Tuple(_rotr90_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rotl90(H::LocalOperator)
    lattice2 = rotl90(H.lattice)
    terms2 = (
        (Tuple(_rotl90_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rot180(H::LocalOperator)
    lattice2 = rot180(H.lattice)
    terms2 = (
        (Tuple(_rot180_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end
