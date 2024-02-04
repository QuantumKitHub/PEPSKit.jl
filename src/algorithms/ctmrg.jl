@with_kw struct CTMRG #<: Algorithm
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.tol
    maxiter::Integer = Defaults.maxiter
    miniter::Integer = 4
    verbose::Integer = 0
    fixedspace::Bool = false
end

function MPSKit.leading_boundary(peps::InfinitePEPS, alg::CTMRG, envs=CTMRGEnv(peps))
    return MPSKit.leading_boundary(peps, peps, alg, envs)
end;

#applies left_move, rotates PEPS, applies it again, ... --> convergence
function MPSKit.leading_boundary(
    peps_above::InfinitePEPS,
    peps_below::InfinitePEPS,
    alg::CTMRG,
    envs=CTMRGEnv(peps_above, peps_below),
)
    err = Inf
    iter = 1

    #for convergence criterium we use the on site contracted boundary
    #this convergences, though the value depends on the bond dimension χ
    old_norm = 1.0
    new_norm = old_norm
    ϵ₁ = 1.0
    while (err > alg.tol && iter <= alg.maxiter) || iter <= alg.miniter
        ϵ = 0.0
        for i in 1:4
            envs, ϵ₀ = left_move(peps_above, peps_below, alg, envs)
            ϵ = max(ϵ, ϵ₀)
            envs = rotate_north(envs, EAST)
            peps_above = envs.peps_above
            peps_below = envs.peps_below
        end

        new_norm = contract_ctrmg(envs)

        err = abs(old_norm - new_norm)
        dϵ = abs((ϵ₁ - ϵ) / ϵ₁)
        @ignore_derivatives alg.verbose > 1 && @printf(
            "CTMRG: \titeration: %4d\t\terror: %.2e\t\tnorm: %.10e\t\tϵ: %.2e\t\tdϵ: %.2e\n",
            iter,
            err,
            abs(new_norm),
            ϵ,
            dϵ
        )

        old_norm = new_norm
        ϵ₁ = ϵ
        iter += 1
    end

    #@ignore_derivatives @show iter, new_norm, err
    @ignore_derivatives iter > alg.maxiter &&
        alg.verbose > 0 &&
        @warn "maxiter $(alg.maxiter) reached: error was $(err)"

    return envs
end

#adds two new colums to the left CTM, this leads to new corners and edges that are then renormalized.
function left_move(
    peps_above::InfinitePEPS{PType},
    peps_below::InfinitePEPS{PType},
    alg::CTMRG,
    envs::CTMRGEnv,
) where {PType}
    corners::typeof(envs.corners) = copy(envs.corners)
    edges::typeof(envs.edges) = copy(envs.edges)

    above_projector_type = tensormaptype(spacetype(PType), 1, 3, storagetype(PType))
    below_projector_type = tensormaptype(spacetype(PType), 3, 1, storagetype(PType))
    ϵ = 0.0
    n0 = 1.0
    n1 = 1.0
    for col in 1:size(peps_above, 2)
        cop = mod1(col + 1, size(peps_above, 2))
        com = mod1(col - 1, size(peps_above, 2))

        above_projs = Vector{above_projector_type}(undef, size(peps_above, 1))
        below_projs = Vector{below_projector_type}(undef, size(peps_above, 1))

        # find all projectors
        for row in 1:size(peps_above, 1)
            rop = mod1(row + 1, size(peps_above, 1))
            peps_above_nw = peps_above[row, col]
            peps_above_sw = rotate_north(peps_above[rop, col], WEST)
            peps_below_nw = peps_below[row, col]
            peps_below_sw = rotate_north(peps_below[rop, col], WEST)

            Q2 = northwest_corner(
                envs.edges[WEST, row, col],
                envs.corners[NORTHWEST, row, col],
                envs.edges[NORTH, row, col],
                peps_above_nw,
                peps_below_nw,
            )
            Q1 = northwest_corner(
                envs.edges[SOUTH, mod1(row + 1, end), col],
                envs.corners[SOUTHWEST, mod1(row + 1, end), col],
                envs.edges[WEST, mod1(row + 1, end), col],
                peps_above_sw,
                peps_below_sw,
            )



            trscheme = if alg.fixedspace == true
                truncspace(space(envs.edges[WEST, row, cop], 1))
            else
                alg.trscheme
            end
            #@ignore_derivatives @show norm(Q1*Q2)

            (U, S, V) = tsvd(Q1 * Q2; trunc=trscheme, alg=SVD())

            @ignore_derivatives n0 = norm(Q1 * Q2)^2
            @ignore_derivatives n1 = norm(U * S * V)^2
            @ignore_derivatives ϵ = max(ϵ, (n0 - n1) / n0)

            isqS = sdiag_inv_sqrt(S)

            @planar Q[-1; -2 -3 -4] := isqS[-1; 1] * conj(U[2 3 4; 1]) * Q1[2 3 4; -2 -3 -4]
            @planar P[-1 -2 -3; -4] := Q2[-1 -2 -3; 1 2 3] * conj(V[4; 1 2 3]) * isqS[4; -4]

            @diffset above_projs[row] = Q
            @diffset below_projs[row] = P
        end

        #use the projectors to grow the corners/edges
        for row in 1:size(peps_above, 1)
            Q = above_projs[row]
            P = below_projs[mod1(row - 1, end)]
            rop = mod1(row + 1, size(peps_above, 1))
            rom = mod1(row - 1, size(peps_above, 1))

            @diffset @planar opt=true corners[NORTHWEST, rop, cop][-1; -2] :=
                envs.corners[NORTHWEST, rop, col][1; 2] *
                envs.edges[NORTH, rop, col][2, 3, 4; -2] *
                Q[-1; 1 3 4]
            @diffset @planar opt=true corners[SOUTHWEST, rom, cop][-1; -2] :=
                envs.corners[SOUTHWEST, rom, col][1; 4] *
                envs.edges[SOUTH, rom, col][ -1 2 3 ; 1] *
                P[4 5 6  ;-2 ]*τ[2 6 ; 3 5]
            @diffset @planar opt=true edges[WEST, row, cop][L1 E2 e4;L4] :=
                envs.edges[WEST, row, col][L2 W4 w4; L3]*
                peps_above[row, col][P1; N1 E1 S1 W1]*
                conj(peps_below[row, col][P3; n1 e1 s1 w1])*
                P[L3 N2 n4; L4]*
                Q[L1;L2 S3 s3]*
                τ[w3 W3;W4 w4]*τ[n3 N1;N2 n4]*τ[n2 W2;W3 n3]*τ[n1 w2;w3 n2]*τ[e2 S2;S3 e1]*
                τ[e3 s2;s3 e2]*τ[e4 E1;E2 e3]*τ[S1 s1;s2 S2]*τ[P2 W1;W2 P1]*τ[P3 w1;w2 P2]

        end

        @diffset corners[NORTHWEST, :, cop] ./= norm.(corners[NORTHWEST, :, cop])
        @diffset edges[WEST, :, cop] ./= norm.(edges[WEST, :, cop])
        @diffset corners[SOUTHWEST, :, cop] ./= norm.(corners[SOUTHWEST, :, cop])
    end

    return CTMRGEnv(peps_above, peps_below, corners, edges), ϵ
end

#this makes a "big" corner for the CTMRG algorithm
function northwest_corner(E4, C1, E1, peps_above, peps_below=peps_above)
    @planar opt=true corner[L1 S2 s4;L4 E2 e4] := 
        E4[L1 W2 w2;L2] *
        C1[L2; L3] *
        E1[L3 N3 n3; L4] *
        peps_above[P; N1 E1 S1 W1] *
        conj(peps_below[P; n1 e1 s1 w1])*
        τ[n2 N2;N3 n3]*τ[e3 E1;E2 e4]*τ[e2 N1;N2 e3]*τ[e1 n1;n2 e2]*τ[w2 s2;s1 w1]*τ[W2 s3;s2 W1]*τ[S2 s4;s3 S1]
end

#a usefull error measure for the convergence of the CTMRG algorithm
function contract_ctrmg(
    envs::CTMRGEnv, peps_above=envs.peps_above, peps_below=envs.peps_below
)
    total = 1.0 + 0im

    for r in 1:size(peps_above, 1), c in 1:size(peps_above, 2)
        total *= @planar opt = true  envs.edges[NORTH, r, c][L1 N1 n3;L2] *
            envs.corners[NORTHEAST, r, c][L2; L3] *
            envs.edges[EAST, r, c][L3 E2 e2; L4] *
            envs.corners[SOUTHEAST, r, c][L4; L5] *
            envs.edges[SOUTH, r, c][L5 S4 s4; L6] *
            envs.corners[SOUTHWEST, r, c][L6; L7] *
            envs.edges[WEST, r, c][L7 W3 w5;L8]*
            envs.corners[NORTHWEST, r, c][L8; L1] *
            peps_above[r, c][P1; N1 E1 S1 W1] *
            conj(peps_below[r, c][P5; n1 e1 s1 w1]) *
            τ[w4 W2;W3 w5]*τ[s3 S3;S4 s4]*τ[n2 e1;e2 n1]*τ[n3 E1;E2 n2]*τ[P2 W1;W2 P1]*
            τ[P3 w3;w4 P2]*τ[P4 S2;S3 P3]*τ[P5 s2;s3 P4]*τ[w2 S1;S2 w3]*τ[w1 s1;s2 w2]
        total *= tr(
            envs.corners[NORTHWEST, r, c] *
            envs.corners[NORTHEAST, r, mod1(c - 1, end)] *
            envs.corners[SOUTHEAST, mod1(r - 1, end), mod1(c - 1, end)] *
            envs.corners[SOUTHWEST, mod1(r - 1, end), c],
        )

        total /= @planar opt = true envs.edges[WEST, r, c][L5 EW1 ew1;L6] *
            envs.corners[NORTHWEST, r, c][L6; L1] *
            envs.corners[NORTHEAST, r, mod1(c - 1, end)][L1; L2] *
            envs.edges[EAST, r, mod1(c - 1, end)][L2 EW2 ew2; L3] *
            envs.corners[SOUTHEAST, r, mod1(c - 1, end)][L3; L4] *
            envs.corners[SOUTHWEST, r, c][L4; L5]*
            τ[ew2 EW2;EW1 ew1]

        total /= @planar opt=true envs.corners[NORTHWEST, r, c][L6; L1] *
            envs.edges[NORTH, r, c][L1 NS2 ns2;L2] *
            envs.corners[NORTHEAST, r, c][L2; L3] *
            envs.corners[SOUTHEAST, mod1(r - 1, end), c][L3;L4] *
            envs.edges[SOUTH, mod1(r - 1, end), c][L4 NS1 ns1; L5] *
            envs.corners[SOUTHWEST, mod1(r - 1, end), c][L5; L6] *
            τ[ns2 NS2;NS1 ns1]
    end
    return total
end

#calculate the two site density matrix for some state : ρ = ψ ψ†
function ρ₂_horizontal(r::Int, c::Int, ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    cp = mod1(c + 1, size(ψ, 2))

    @planar opt=true ρ[P5 P10;P1 P6] :=
        env.edges[NORTH, r, c][L1 N2 n1; L2] *
        env.edges[NORTH, r, cp][L3 N4 n2; L4] *
        env.corners[NORTHEAST, r, cp][L5; L6] *
        env.edges[EAST, r, cp][L6 E3 e4; L7] *
        env.corners[SOUTHEAST, r, cp][L7; L8]*
        env.edges[SOUTH, r, cp][L8 S4 s10;L9] *
        env.edges[SOUTH, r, c][L10 S2 s5;L11] *
        env.corners[SOUTHWEST, r, c][L12 ; L13] *
        env.edges[WEST, r, c][L13 W2 w4;L14] *
        env.corners[NORTHWEST, r, c][L14; L1] *
        ψ[r, c][P3;N1 H1 S1 W1] *
        ψ[r, cp][P8; N3 E1 S3 H3] *
        conj(ψ[r, c][P2;n1 h1 s1 w1]) *
        conj(ψ[r, cp][P7; n2 e1 s6 h6]) *
        τ[w4 N1;N2 w3]*τ[P4 W1;W2 P3]*τ[L12 P5;P4 L11]*τ[P2 L3;L2 P1]*τ[w2 s2;s1 w1]*
        τ[w3 h2;h1 w2]*τ[s3 h3;h2 s2]*τ[H1 s4;s3 H2]*τ[s5 S2;S1 s4]*τ[N3 h4;h3 N4]*
        τ[L10 P10;P9 L9]*τ[h4 e2;e1 h5]*τ[E1 e3;e2 E2]*τ[P9 H3;H2 P8]*τ[s10 S4;S3 s9]*
        τ[s9 e4;e3 s8]*τ[s8 E3; E2 s7]*τ[h5 s7;s6 h6]*τ[P7 L5;L4 P6]
    return ρ
end

