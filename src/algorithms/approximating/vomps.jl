function vomps(mps,pepst,init,trscheme;tol=1e-12,maxit = 400)
    #assume init to be rightorthed
    leftstart = permute(TensorMap(I,eltype(mps[1]),space(init[1],1)*space(pepst[1],1)',space(pepst[1],1)'*space(mps[1],1)),(1,2,3),(4,));
    rightstart = permute(TensorMap(I,eltype(mps[1]),space(mps[end],4)'*space(pepst[end],3)',space(pepst[end],3)'*space(init[end],4)'),(1,2,3),(4,));

    GL = [leftstart];GR = [rightstart];
    for i in 1:(length(mps)-1)
        push!(GL,mps_apply_transfer_left(GL[end],pepst[i],mps[i],init[i]));
        push!(GR,mps_apply_transfer_right(GR[end],pepst[end-i+1],mps[end-i+1],init[end-i+1]));
    end
    reverse!(GR);

    err = 0.0;
    for maxit = 1:100

        err = 0.0;
        for i = 1:length(mps)-1
            @tensor temp[-1 -2 -3;-4]:= GL[i][-1,7,8,9]*mps[i][9,5,3,1]*GR[i][1,4,2,-4]*pepst[i][7,-2,4,5,6]*conj(pepst[i][8,-3,2,3,6])
            err = max(err,norm(temp-init[i],Inf)/(norm(temp,Inf)))

            (init[i],c) = TensorKit.leftorth(temp);
            @tensor init[i+1][-1 -2 -3;-4]:=c[-1,1]*init[i+1][1,-2,-3,-4]

            GL[i+1] = mps_apply_transfer_left(GL[i],pepst[i],mps[i],init[i])
        end

        for i = length(mps):-1:2
            @tensor temp[-1; -2 -3 -4]:= GL[i][-1,7,8,9]*mps[i][9,5,3,1]*GR[i][1,4,2,-4]*pepst[i][7,-2,4,5,6]*conj(pepst[i][8,-3,2,3,6])
            err = max(err,norm(temp-permute(init[i],(1,),(2,3,4)),Inf)/(norm(temp,Inf)))

            (c,temp) = TensorKit.rightorth(temp);
            init[i] = permute(temp,(1,2,3),(4,))
            init[i-1]=init[i-1]*c

            GR[i-1] = mps_apply_transfer_right(GR[i],pepst[i],mps[i],init[i])
        end

        if err < tol
            break
        end

    end

    err > tol && @info "vomps failed to converge $(err)"
    #init = truncatebonds(FiniteMps(init),trscheme).data;

    return init
end

function vomps(mps,pepsl#=horizonal=#,pepsc#=special=#,pepsr#=vertical=#,init,trscheme)
    #println("cvomps")
    LPspace = space([pepsl;pepsc][1],1);
    RPspace = space([pepsc;pepsr][end],4);

    #assume init to be rightorthed
    leftstart = permute(TensorMap(I,eltype(init[1]),space(init[1],1)*LPspace',LPspace'*space(init[1],1)),(1,2,3),(4,));
    rightstart = permute(TensorMap(I,eltype(init[1]),space(init[end],4)'*RPspace',RPspace'*space(init[end],4)'),(1,2,3),(4,));

    GL = [leftstart];GR = [rightstart];
    for i in 1:(length(mps))
        if i<=length(pepsl)
            push!(GL,mps_apply_transfer_left(GL[end],pepsl[i],mps[i],init[i]));
        elseif i == length(pepsl)+1
            @tensor trans[-1 -2 -3;-4]:=GL[end][7,8,9,-4]*pepsc[8,4,5,-2,6]*conj(pepsc[9,2,3,-3,6])*conj(init[i][7,4,2,1])*conj(init[i+1][1,5,3,-1])
            push!(GL,trans);
        else
            push!(GL,mps_apply_transfer_left(GL[end],rotate_north(pepsr[i-length(pepsl)-1],West),mps[i-1],init[i+1]));
        end

        if i<=length(pepsr)
            push!(GR,mps_apply_transfer_right(GR[end],rotate_north(pepsr[end-i+1],West),mps[end-i+1],init[end-i+1]));
        elseif i == length(pepsr)+1
            @tensor trans[-1 -2 -3;-4]:=GR[end][-1,7,8,9]*pepsc[-2,4,5,7,6]*conj(pepsc[-3,2,3,8,6])*conj(init[end-i+1-1][-4,4,2,1])*conj(init[end-i+1][1,5,3,9])
            push!(GR,trans)
        else
            push!(GR,mps_apply_transfer_right(GR[end],pepsl[end-(i-length(pepsr)-1)+1],mps[end-i+1+1],init[end-i+1-1]));
        end
    end
    reverse!(GR);


    tol = 1e-12
    err = 0.0;
    for maxit = 1:100

        err = 0.0;
        for i = 1:length(init)-1
            if i<=length(pepsl)
                @tensor temp[-1 -2 -3;-4]:= GL[i][-1,7,8,9]*mps[i][9,5,3,1]*GR[i][1,4,2,-4]*pepsl[i][7,-2,4,5,6]*conj(pepsl[i][8,-3,2,3,6])
            elseif i == length(pepsl)+1
                @tensor temp[-1 -2 -3;-4]:= GL[i][-1,2,4,1]*GR[i][1,3,5,9]*pepsc[2,-2,7,3,6]*conj(pepsc[4,-3,8,5,6])*conj(init[i+1][-4,7,8,9])
            elseif i == length(pepsl)+2
                @tensor temp[-1 -2 -3;-4]:= GL[i-1][7,2,4,1]*GR[i-1][1,3,5,-4]*pepsc[2,8,-2,3,6]*conj(pepsc[4,9,-3,5,6])*conj(init[i-1][7,8,9,-1])
            else
                @tensor temp[-1 -2 -3;-4]:= GL[i-1][-1,8,9,1]*mps[i-2][1,5,6,7]*GR[i-1][7,3,2,-4]*pepsr[i-2-length(pepsl)][5,8,-2,3,4]*conj(pepsr[i-2-length(pepsl)][6,9,-3,2,4])
            end

            err = max(err,norm(temp-init[i],Inf)/(norm(temp,Inf)+1.0))

            (init[i],c) = TensorKit.leftorth(temp);
            @tensor init[i+1][-1 -2 -3;-4]:=c[-1,1]*init[i+1][1,-2,-3,-4]

            if i<=length(pepsl)
                GL[i+1] = mps_apply_transfer_left(GL[i],pepsl[i],mps[i],init[i]);
            elseif i == length(pepsl)+1 #don't do anything
            elseif i == length(pepsl)+2
                @tensor trans[-1 -2 -3;-4]:=GL[i-1][7,8,9,-4]*pepsc[8,4,5,-2,6]*conj(pepsc[9,2,3,-3,6])*conj(init[i-1][7,4,2,1])*conj(init[i][1,5,3,-1])
                GL[i] = trans
            else
                GL[i] = mps_apply_transfer_left(GL[i-1],rotate_north(pepsr[i-length(pepsl)-2],West),mps[i-2],init[i]);
            end
        end

        for i = length(init):-1:2
            if i<=length(pepsl)
                @tensor temp[-1; -2 -3 -4]:= GL[i][-1,7,8,9]*mps[i][9,5,3,1]*GR[i][1,4,2,-4]*pepsl[i][7,-2,4,5,6]*conj(pepsl[i][8,-3,2,3,6])
            elseif i == length(pepsl)+1
                @tensor temp[-1; -2 -3 -4]:= GL[i][-1,2,4,1]*GR[i][1,3,5,9]*pepsc[2,-2,7,3,6]*conj(pepsc[4,-3,8,5,6])*conj(init[i+1][-4,7,8,9])
            elseif i == length(pepsl)+2
                @tensor temp[-1; -2 -3 -4]:= GL[i-1][7,2,4,1]*GR[i-1][1,3,5,-4]*pepsc[2,8,-2,3,6]*conj(pepsc[4,9,-3,5,6])*conj(init[i-1][7,8,9,-1])
            else
                @tensor temp[-1; -2 -3 -4]:= GL[i-1][-1,8,9,1]*mps[i-2][1,5,6,7]*GR[i-1][7,3,2,-4]*pepsr[i-2-length(pepsl)][5,8,-2,3,4]*conj(pepsr[i-2-length(pepsl)][6,9,-3,2,4])
            end
            err = max(err,norm(temp-permute(init[i],(1,),(2,3,4)),Inf)/(norm(temp,Inf)+1.0))

            (c,temp) = TensorKit.rightorth(temp);
            init[i] = permute(temp,(1,2,3),(4,))
            init[i-1]=init[i-1]*c


            if i<=length(pepsl)
                GR[i-1]=mps_apply_transfer_right(GR[i],pepsl[i],mps[i],init[i]);
            elseif i == length(pepsl)+1
                @tensor trans[-1 -2 -3;-4]:=GR[i][-1,7,8,9]*pepsc[-2,4,5,7,6]*conj(pepsc[-3,2,3,8,6])*conj(init[i][-4,4,2,1])*conj(init[i+1][1,5,3,9])
                GR[i-1] = trans
            elseif i == length(pepsl)+2 #don't do anything

            else
                GR[i-2] = mps_apply_transfer_right(GR[i-1],rotate_north(pepsr[i-length(pepsl)-2],West),mps[i-2],init[i]);
            end

        end

        #@show err
        if err < tol
            break
        end

    end
    err > tol && @info "cornervomps failed to converge $(err)"
    return init
end
