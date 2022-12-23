@with_kw struct CTMRG #<: Algorithm
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.tol
    maxiter::Integer = Defaults.maxiter
    miniter::Integer = 4
    verbose::Integer = 0
    fixedspace::Bool = false
end

@with_kw struct CTMRG2 #<: Algorithm
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.tol
    maxiter::Integer = Defaults.maxiter
    miniter::Integer = 4
    verbose::Integer = 0
end

MPSKit.leading_boundary(peps::InfinitePEPS,alg::CTMRG,envs = CTMRGEnv(peps)) = MPSKit.leading_boundary(peps,peps,alg,envs);

function MPSKit.leading_boundary(peps_above::InfinitePEPS,peps_below::InfinitePEPS,alg::CTMRG,envs = CTMRGEnv(peps_above,peps_below))
    err = Inf
    iter = 1

    #for convergence criterium we use the on site contracted boundary
    #this convergences, though the value depends on the bond dimension χ
    old_norm = 1.0
    new_norm = old_norm
    ϵ₁ = 1.0
    while (err>alg.tol&&iter<=alg.maxiter) || iter<=alg.miniter
        ϵ = 0.0
        for i in 1:4
            envs,ϵ₀ = left_move(peps_above, peps_below,alg,envs);
            ϵ = max(ϵ,ϵ₀)
            envs = rotate_north(envs,EAST);
            peps_above = envs.peps_above;
            peps_below = envs.peps_below;
        end
        
        new_norm = contract_ctrmg(envs)

        err = abs(old_norm-new_norm)
        dϵ = abs((ϵ₁-ϵ)/ϵ₁)
        @ignore_derivatives alg.verbose > 1 && @printf("%4d   %.2e   %.10e   %.2e    %.2e\n",
         iter,err,abs(new_norm),ϵ,dϵ)

        old_norm = new_norm
        ϵ₁ = ϵ
        iter += 1
    end

    #@ignore_derivatives @show iter, new_norm, err
    @ignore_derivatives iter > alg.maxiter && alg.verbose > 0 && @warn "maxiter $(alg.maxiter) reached: error was $(err)"

    return envs
end

function left_move(peps_above::InfinitePEPS{PType},peps_below::InfinitePEPS{PType},alg::CTMRG,envs::CTMRGEnv) where PType
    corners::typeof(envs.corners) = copy(envs.corners);
    edges::typeof(envs.edges) = copy(envs.edges);

    above_projector_type = tensormaptype(spacetype(PType),1,3,storagetype(PType));
    below_projector_type = tensormaptype(spacetype(PType),3,1,storagetype(PType));
    ϵ = 0.0
    n0 = 1.0
    n1 = 1.0
    for col in 1:size(peps_above,2)
        cop = mod1(col+1,size(peps_above,2))
        com = mod1(col-1,size(peps_above,2))

        above_projs = Vector{above_projector_type}(undef,size(peps_above,1));
        below_projs = Vector{below_projector_type}(undef,size(peps_above,1));

        # find all projectors
        for row in 1:size(peps_above,1)
            rop = mod1(row+1, size(peps_above,1))
            peps_above_nw = peps_above[row,col];
            peps_above_sw = rotate_north(peps_above[rop,col],WEST);
            peps_below_nw = peps_below[row,col];
            peps_below_sw = rotate_north(peps_below[rop,col],WEST);

            Q1 = northwest_corner(envs.edges[SOUTH,mod1(row+1,end),col],envs.corners[SOUTHWEST,mod1(row+1,end),col],envs.edges[WEST,mod1(row+1,end),col],peps_above_sw,peps_below_sw);
            Q2 = northwest_corner(envs.edges[WEST,row,col],envs.corners[NORTHWEST,row,col],envs.edges[NORTH,row,col],peps_above_nw,peps_below_nw);


            trscheme = alg.fixedspace == true ? truncspace(space(envs.edges[WEST,row,cop],1)) : alg.trscheme
            #@ignore_derivatives @show norm(Q1*Q2)
            
            (U,S,V) = tsvd(Q1*Q2,trunc = trscheme,alg = SVD())
            
            @ignore_derivatives n0 = norm(Q1*Q2)^2
            @ignore_derivatives n1 = norm(U*S*V)^2
            @ignore_derivatives ϵ = max(ϵ, (n0-n1)/n0)

            isqS = sdiag_inv_sqrt(S);
            #Q = isqS*U'*Q1;
            #P = Q2*V'*isqS;
            @tensor Q[-1;-2 -3 -4] := isqS[-1;1] * conj(U[2 3 4;1]) * Q1[2 3 4;-2 -3 -4]
            @tensor P[-1 -2 -3;-4] := Q2[-1 -2 -3;1 2 3] * conj(V[4;1 2 3]) * isqS[4;-4]

            @diffset above_projs[row] = Q;
            @diffset below_projs[row] = P;
        end
        
        
        #use the projectors to grow the corners/edges
        for row in 1:size(peps_above,1)
            Q = above_projs[row];
            P = below_projs[mod1(row-1,end)];
            rop = mod1(row+1,size(peps_above,1))
            rom = mod1(row-1,size(peps_above,1))            
            
            @diffset @tensor corners[NORTHWEST,rop,cop][-1;-2] := envs.corners[NORTHWEST,rop,col][1,2] * envs.edges[NORTH,rop,col][2,3,4,-2]*Q[-1;1 3 4]
            @diffset @tensor corners[SOUTHWEST,rom,cop][-1;-2] := envs.corners[SOUTHWEST,rom,col][1,4] * envs.edges[SOUTH,rom,col][-1,2,3,1]*P[4 2 3;-2]
            @diffset @tensor edges[WEST,row,cop][-1 -2 -3;-4] := envs.edges[WEST,row,col][1 2 3;4]*
            peps_above[row,col][9;5 -2 7 2]*
            conj(peps_below[row,col][9;6 -3 8 3])*
            P[4 5 6;-4]*
            Q[-1;1 7 8]
        end


        @diffset corners[NORTHWEST,:,cop]./=norm.(corners[NORTHWEST,:,cop]);
        @diffset edges[WEST,:,cop]./=norm.(edges[WEST,:,cop]);
        @diffset corners[SOUTHWEST,:,cop]./=norm.(corners[SOUTHWEST,:,cop]);
    end
    
    
    return CTMRGEnv(peps_above,peps_below,corners,edges), ϵ
end

northwest_corner(E4,C1,E1,peps_above,peps_below=peps_above) = @tensor corner[-1 -2 -3;-4 -5 -6] := E4[-1 1 2;3]*C1[3;4]*E1[4 5 6;-4]*peps_above[7;5 -5 -2 1]*conj(peps_below[7;6 -6 -3 2])
northeast_corner(E1,C2,E2,peps_above,peps_below=peps_above) = @tensor corner[-1 -2 -3;-4 -5 -6] := E1[-1 1 2;3]*C2[3;4]*E2[4 5 6;-4]*peps_above[7;1 5 -5 -2]*conj(peps_below[7;2 6 -6 -3])
southeast_corner(E2,C3,E3,peps_above,peps_below=peps_above) = @tensor corner[-1 -2 -3;-4 -5 -6] := E2[-1 1 2;3]*C3[3;4]*E3[4 5 6;-4]*peps_above[7;-2 1 5 -5]*conj(peps_below[7;-3 2 6 -6])

#=

function MPSKit.leading_boundary(peps::InfinitePEPS,alg::CTMRG2,envs = CTMRGEnv(peps))
    err = Inf
    iter = 1

    old_norm = 1.0

    while (err > alg.tol && iter <= alg.maxiter) || iter<4

        for dir in 1:4
            envs = left_move(peps,alg,envs);

            envs = rotate_north(envs,EAST);
            peps = envs.peps;
        end
        new_norm = abs(contract_ctrmg(peps,envs,1,1))
        #@show new_norm
        err = abs(old_norm-new_norm)
        @ignore_derivatives mod(alg.verbose,alg.miniter)==0 && mod(iter,alg.verbose+1)==0 &&  @info "$(iter) $(err) $(new_norm)"
        

        old_norm = new_norm
        iter += 1
    end
    @ignore_derivatives iter > alg.maxiter && @warn "maxiter $(alg.maxiter) reached: error was $(err)"

    envs
end

# the actual left_move is dependent on the type of ctmrg, so this seems natural
function left_move(peps::InfinitePEPS{PType},alg::CTMRG2,envs::CTMRGEnv) where PType
    corners::typeof(envs.corners) = copy(envs.corners);
    edges::typeof(envs.edges) = copy(envs.edges);

    above_projector_type = tensormaptype(spacetype(PType),1,3,storagetype(PType));
    below_projector_type = tensormaptype(spacetype(PType),3,1,storagetype(PType));

    for col in 1:size(peps,2)
        cop = mod1(col+1, size(peps,2))
        above_projs = Vector{above_projector_type}(undef,size(peps,1));
        below_projs = Vector{below_projector_type}(undef,size(peps,1));

        # find all projectors
        for row in 1:size(peps,1)
            rop = mod1(row+1, size(peps,1))
            peps_nw = peps[row,col];
            peps_sw = rotate_north(peps[rop,col],WEST);

            Q1 = northwest_corner(envs.edges[WEST,row,col],envs.corners[NORTHWEST,row,col],  envs.edges[NORTH,row,col],peps_nw);
            Q2 = northeast_corner(envs.edges[NORTH,row,cop],envs.corners[NORTHEAST,row,cop],envs.edges[EAST,row,cop],peps[row,cop])
            Q3 = southeast_corner(envs.edges[EAST,rop,cop],envs.corners[SOUTHEAST,rop,cop],envs.edges[SOUTH,rop,cop],peps[rop,cop])
            Q4 = northwest_corner(envs.edges[SOUTH,rop,col],envs.corners[SOUTHWEST,rop,col],envs.edges[WEST,rop,col],peps_sw);
            Qnorth = Q1*Q2
            Qsouth = Q3*Q4
            (U,S,V) = tsvd(Qsouth*Qnorth, trunc = alg.trscheme);
            #@ignore_derivatives @show ϵ = real(norm(Qsouth*Qnorth)^2-norm(U*S*V)^2) 
            #@ignore_derivatives @info ϵ
            isqS = sdiag_inv_sqrt(S);
            Q = isqS*U'*Qsouth;
            P = Qnorth*V'*isqS;

            @diffset above_projs[row] = Q;
            @diffset below_projs[row] = P;
        end

        #use the projectors to grow the corners/edges
        for row in 1:size(peps,1)
            Q = above_projs[row];
            P = below_projs[mod1(row-1,end)];

            @diffset @tensor corners[NORTHWEST,row+1,col+1][-1;-2] := envs.corners[NORTHWEST,row+1,col][1,2] * envs.edges[NORTH,row+1,col][2,3,4,-2]*Q[-1;1 3 4]
            @diffset @tensor corners[SOUTHWEST,row-1,col+1][-1;-2] := envs.corners[SOUTHWEST,row-1,col][1,4] * envs.edges[SOUTH,row-1,col][-1,2,3,1]*P[4 2 3;-2]
            @diffset @tensor edges[WEST,row,col+1][-1 -2 -3;-4] := envs.edges[WEST,row,col][1 2 3;4]*
            peps[row,col][9;5 -2 7 2]*
            conj(peps[row,col][9;6 -3 8 3])*
            P[4 5 6;-4]*
            Q[-1;1 7 8]
        end

        @diffset corners[NORTHWEST,:,col+1]./=norm.(corners[NORTHWEST,:,col+1]);
        @diffset corners[SOUTHWEST,:,col+1]./=norm.(corners[SOUTHWEST,:,col+1]);
        @diffset edges[WEST,:,col+1]./=norm.(edges[WEST,:,col+1]);
    end
    
    CTMRGEnv(peps,corners,edges);
end
=#

function contract_ctrmg(envs::CTMRGEnv,peps_above = envs.peps_above, peps_below = envs.peps_below)
    total = 1.0+0im;

    for r in 1:size(peps_above,1), c in 1:size(peps_above,2) 
        total*=@tensor envs.edges[WEST,r,c][1 2 3;4]*
            envs.corners[NORTHWEST,r,c][4;5]*
            envs.edges[NORTH,r,c][5 6 7;8]*
            envs.corners[NORTHEAST,r,c][8;9]*
            envs.edges[EAST,r,c][9 10 11;12]*
            envs.corners[SOUTHEAST,r,c][12;13]*
            envs.edges[SOUTH,r,c][13 14 15;16]*
            envs.corners[SOUTHWEST,r,c][16;1]*
            peps_above[r,c][17;6 10 14 2]*
            conj(peps_below[r,c][17;7 11 15 3])
        total *= tr(envs.corners[NORTHWEST,r,c]*envs.corners[NORTHEAST,r,mod1(c-1,end)]*envs.corners[SOUTHEAST,mod1(r-1,end),mod1(c-1,end)]*envs.corners[SOUTHWEST,mod1(r-1,end),c])
        
        total /= @tensor envs.edges[WEST,r,c][1 10 11;4]*
            envs.corners[NORTHWEST,r,c][4;5]*
            envs.corners[NORTHEAST,r,mod1(c-1,end)][5;6]*
            envs.edges[EAST,r,mod1(c-1,end)][6 10 11;7]*
            envs.corners[SOUTHEAST,r,mod1(c-1,end)][7;8]*
            envs.corners[SOUTHWEST,r,c][8;1]

        total /= @tensor envs.corners[NORTHWEST,r,c][1;2]*
            envs.edges[NORTH,r,c][2 10 11;3]*
            envs.corners[NORTHEAST,r,c][3;4]*
            envs.corners[SOUTHEAST,mod1(r-1,end),c][4;5]*
            envs.edges[SOUTH,mod1(r-1,end),c][5 10 11;6]*
            envs.corners[SOUTHWEST,mod1(r-1,end),c][6;1]

    end
    
    total
end

