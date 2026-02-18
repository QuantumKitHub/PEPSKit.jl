#TODO: Add docs and figure
struct CTMRGTriaEnv{T}
    "6 x rows x cols array of corner C tensors, where the first dimension specifies the spatial direction"
    C::Array{T, 3}
    "6 x rows x cols array of edge Ta tensors, where the first dimension specifies the spatial direction"
    Ea::Array{T, 3}
    "6 x rows x cols array of edge Ta tensors, where the first dimension specifies the spatial direction"
    Eb::Array{T, 3}
end
# function CTMRGTriaEnv(corners::Array{C, 3}, edges::Array{T, 3}) where {C, T}
#     foreach(check_environment_virtualspace, edges)
#     return CTMRGTriaEnv{C, T}(corners, edges)
# end

# """
#     CTMRGEnv(
#         [f=randn, T=ComplexF64], Ds_north::A, Ds_east::A, chis_north::B, [chis_east::B], [chis_south::B], [chis_west::B]
#     ) where {A<:AbstractMatrix{<:VectorSpace}, B<:AbstractMatrix{<:ElementarySpace}}

# Construct a CTMRG environment by specifying matrices of north and east virtual spaces of the
# corresponding partition function and the north, east, south and west virtual spaces of the
# environment. Each respective matrix entry corresponds to a site in the unit cell. By
# default, the virtual environment spaces for all directions are taken to be the same.

# The environment virtual spaces for each site correspond to the north or east virtual space
# of the corresponding edge tensor for each direction. Specifically, for a given site
# `(r, c)`, `chis_north[r, c]` corresponds to the east space of the north edge tensor,
# `chis_east[r, c]` corresponds to the north space of the east edge tensor,
# `chis_south[r, c]` corresponds to the east space of the south edge tensor, and
# `chis_west[r, c]` corresponds to the north space of the west edge tensor.

# Each entry of the `Ds_north` and `Ds_east` matrices corresponds to an effective local space
# of the partition function, and can be represented as an `ElementarySpace` (e.g. for the case
# of a partition function defined in terms of local rank-4 tensors) or a `ProductSpace` (e.g.
# for the case of a network representing overlaps of PEPSs and PEPOs).
# """
function get_Ds(D::A) where {A <: ProductSpace}
    return [dir > 3 ? reverse(D') : D for dir in 1:6]
end

function get_Ds(D::A) where {A <: ElementarySpace}
    return [dir > 3 ? D' : D for dir in 1:6]
end

function CTMRGTriaEnv(
        f, T, D::Union{A, B}, chis::B; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {
        A <: ProductSpace, B <: ElementarySpace,
    }
    if typeof(D) <: ElementarySpace
        N = 1
    else
        N = length(D)
    end
    st = spacetype(D)
    T_type = tensormaptype(st, N + 1, 1, T)

    Cs = Array{T_type}(undef, 6, unitcell...)
    Eas = Array{T_type}(undef, 6, unitcell...)
    Ebs = Array{T_type}(undef, 6, unitcell...)

    Ds = get_Ds(D)
    for dir in 1:6, r in 1:unitcell[1], c in 1:unitcell[2]
        C = _edge_tensor(f, T, chis, Ds[dir], chis)
        Ea = _edge_tensor(f, T, chis, Ds[dir], chis)
        Eb = _edge_tensor(f, T, chis, Ds[mod1(dir + 1, 6)], chis)

        C /= norm(C)
        Ea /= norm(Ea)
        Eb /= norm(Eb)

        Cs[dir, r, c] = C
        Eas[dir, r, c] = Ea
        Ebs[dir, r, c] = Eb
    end
    return CTMRGTriaEnv(Cs, Eas, Ebs)
end
