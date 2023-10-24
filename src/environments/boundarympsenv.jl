# Some form of boundary MPS environments for infinite PEPS and PEPO routines

## Utility

algtol(alg::VUMPS) = alg.tol_galerkin
algtol(alg::GradientGrassmann) = alg.method.gradtol
update_tol(alg::VUMPS, tol) = @set alg.tol_galerkin = tol
function update_tol(alg::GradientGrassmann, tol) # annoying disparity between typedef and actual constructor...
    m = alg.method
    m = @set m.gradtol = tol
    return GradientGrassmann(; method=m, (finalize!)=alg.finalize!)
end

## Boundary MPS environment manager

mutable struct BoundaryMPSEnv{A,E,F} <: Cache
    boundaries::A
    envs::E
    alg::F
end

## PEPS boundary MPS

function MPSKit.environments(
    peps::InfinitePEPS,
    alg::A=VUMPS();
    vspaces=[oneunit(space(peps, 1, 1))],
    hermitian=false,
    kwargs...,
) where {A<:Union{VUMPS,GradientGrassmann}}
    tr = TransferPEPSMultiline(peps, 1)
    return environments(tr, alg; vspaces, hermitian, kwargs...)
end

function MPSKit.recalculate!(
    envs::BoundaryMPSEnv,
    peps::InfinitePEPS;
    tol=algtol(envs.alg),
    hermitian=false,
    kwargs...,
)
    tr = TransferPEPSMultiline(peps, 1)
    return recalculate!(envs, tr; tol, hermitian, kwargs...)
end

# this probably needs a different name?
# computes norm-per-site of PEPS-PEPS overlap using above, below and mixed environments
function MPSKit.expectation_value(peps::InfinitePEPS, ca::BoundaryMPSEnv)
    retval = PeriodicArray{scalartype(peps),2}(undef, size(peps)...)
    above = ca.boundaries[1]
    below = ca.boundaries[2]
    (lw, rw) = ca.envs[3]
    for (row, col) in product(1:size(peps, 1), 1:size(peps, 2))
        fliprow = size(peps, 1) - row + 1 # below starts counting from below
        retval[row, col] = @tensor lw[row, col][1 2 4; 7] *
            conj(below.AC[fliprow, col][1 3 6; 13]) *
            peps[row, col][5; 8 11 3 2] *
            conj(peps[row, col][5; 9 12 6 4]) *
            above.AC[row, col][7 8 9; 10] *
            rw[row, col][10 11 12; 13]
    end
    return retval
end

function MPSKit.normalize!(peps::InfinitePEPS, envs::BoundaryMPSEnv)
    norm_per_site = expectation_value(peps, envs)
    for (row, col) in product(1:size(peps, 1), 1:size(peps, 2))
        scale!(peps[row, col], 1 / sqrt(norm_per_site[row, col]))
    end
end

# naming convention? this does not return an effective operator, so naming is probably misleading
function ∂∂peps(peps::InfinitePEPS{T}, ca::BoundaryMPSEnv) where {T<:PEPSTensor}
    retval = PeriodicArray{T,2}(undef, size(peps)...)
    above = ca.boundaries[1]
    below = ca.boundaries[2]
    (lw, rw) = ca.envs[3]
    for (row, col) in product(1:size(peps, 1), 1:size(peps, 2))
        fliprow = size(peps, 1) - row + 1 # below starts counting from below
        retval[row, col] = ∂peps(
            above.AC[row, col],
            below.AC[fliprow, col],
            peps[row, col],
            lw[row, col],
            rw[row, col],
        )
    end
    return retval
end

## PEPO boundary MPS

function MPSKit.environments(
    peps::InfinitePEPS,
    pepo::InfinitePEPO,
    alg::A=VUMPS();
    vspaces=[oneunit(space(peps, 1, 1))],
    hermitian=false,
    kwargs...,
) where {A<:Union{VUMPS,GradientGrassmann}}
    tr = TransferPEPOMultiline(peps, pepo, 1)
    return environments(tr, alg; vspaces, hermitian, kwargs...)
end

function MPSKit.recalculate!(
    envs::BoundaryMPSEnv,
    peps::InfinitePEPS,
    pepo::InfinitePEPO;
    tol=algtol(envs.alg),
    hermitian=false,
    kwargs...,
)
    tr = TransferPEPOMultiline(peps, pepo, 1)
    return recalculate!(envs, tr; tol, hermitian, kwargs...)
end

# this probably needs a different name?
# computes norm-per-site of PEPS-PEPO-PEPS sandwich using above, below and mixed environments
function MPSKit.expectation_value(
    peps::InfinitePEPS, pepo::InfinitePEPO, ca::BoundaryMPSEnv
)
    retval = PeriodicArray{scalartype(peps),2}(undef, size(peps)...)
    opp = TransferPEPOMultiline(peps, pepo, 1) # for convenience...
    N = height(opp[1]) + 4
    above = ca.boundaries[1]
    below = ca.boundaries[2]
    (lw, rw) = ca.envs[3]
    for (row, col) in product(1:size(peps, 1), 1:size(peps, 2))
        fliprow = size(peps, 1) - row + 1 # below starts counting from below
        O_rc = opp[row, col]
        GL´ = transfer_left(lw[row, col], O_rc, above.AC[row, col], below.AC[fliprow, col])
        retval[row, col] = TensorKit.TensorOperations.tensorscalar(
            ncon([GL´, rw[row, col]], [[N, (2:(N - 1))..., 1], [(1:N)...]])
        )
    end
    return retval
end

# naming convention? this does not return an effective operator, so naming is probably misleading
function ∂∂peps(
    peps::InfinitePEPS{T}, pepo::InfinitePEPO, ca::BoundaryMPSEnv
) where {T<:PEPSTensor}
    retval = PeriodicArray{T,2}(undef, size(peps)...)
    opp = TransferPEPOMultiline(peps, pepo, 1) # just for convenience
    above = ca.boundaries[1]
    below = ca.boundaries[2]
    (lw, rw) = ca.envs[3]
    for (row, col) in product(1:size(peps, 1), 1:size(peps, 2))
        fliprow = size(peps, 1) - row + 1 # below starts counting from below
        O_rc = opp[row, col]
        retval[row, col] = ∂peps(
            above.AC[row, col],
            below.AC[fliprow, col],
            (O_rc[1], O_rc[3]),
            lw[row, col],
            rw[row, col],
        )
    end
    return retval
end

## The actual routines

# because below starts counting from below
_flippedy(M::MPSMultiline) = MPSKit.Multiline(circshift(reverse(M.data), 1))

function MPSKit.environments(
    tr::Union{TransferPEPSMultiline,TransferPEPOMultiline},
    alg::A=VUMPS();
    vspaces=[oneunit(space(tr, 1, 1))], # defaults to trivial
    hermitian=false,
) where {A<:Union{VUMPS,GradientGrassmann}}
    # above boundary
    above = initializeMPS(tr, vspaces)
    envs_above = environments(above, tr)

    # maybe below boundary
    if hermitian # literally the same
        below = above
        envs_below = envs_above
        envs_mixed = (envs_above.lw, envs_above.rw)
    else
        error("not implemented yet")
        tr_dag = dagger(tr)
        below = initializeMPS(tr_dag, vspaces)
        envs_below = envirlnments(below, tr_dag)

        # mixed environments
        envs_mixed = MPSKit.mixed_fixpoints(above, tr, _flippedy(below))
    end

    # collect
    boundaries = [above, below]
    envs = [envs_above, envs_below, envs_mixed]

    return recalculate!(BoundaryMPSEnv(boundaries, envs, alg), tr; hermitian)
end

function MPSKit.recalculate!(
    envs::BoundaryMPSEnv,
    tr::Union{TransferPEPSMultiline,TransferPEPOMultiline};
    tol=algtol(envs.alg),
    hermitian=false,
)
    alg = algtol(envs.alg) == tol ? envs.alg : update_tol(envs.alg, tol)

    # above boundary
    envs.envs[1].opp = tr # because pre-supplied environments only ever use their own operator...
    envs.boundaries[1], envs.envs[1], err = leading_boundary(
        envs.boundaries[1], tr, alg, envs.envs[1]
    )

    # maybe below boundary
    if hermitian # literally the same
        envs.boundaries[2], envs.envs[2] = envs.boundaries[1], envs.envs[1]
        envs.envs[3] = (envs.envs[1].lw, envs.envs[1].rw)
    else
        error("not implemented yet")
        tr_dag = dagger(tr) # TODO: actually define this...
        envs.envs[2].opp = tr_dag # because pre-supplied environments only ever use their own operator...
        envs.boundaries[2], envs.envs[2], err = leading_boundary(
            envs.boundaries[2], tr_dag, alg, envs.envs[2]
        )
        # mixed environments
        envs.envs[3] = MPSKit.mixed_fixpoints(
            envs.boundaries[1], tr, _flippedy(envs.boundaries[2]), envs.envs[3]
        )
    end

    envs.alg = alg

    return envs
end

## Channel MPS environments

# TODO: actually try this again at some point?
mutable struct ChannelMPSEnv{A,C,F} <: Cache
    boundaries::A
    corners::C
    alg::F
end
