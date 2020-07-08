#=
contains some utility methods to work with envmanager objects
=#
AL(man,dir,row::Colon,col::Int) = AL(man,dir,1:size(man.peps,1),col)
AL(man,dir::Dir,row::UnitRange{Int64},col::Int) = [AL(man,dir,r,col) for r in row];
AL(man,dir,row::Int,col::Colon) = AL(man,dir,row,1:size(man.peps,2))
AL(man,dir::Dir,row::Int,col::UnitRange{Int64}) = [AL(man,dir,row,c) for c in col];

AR(man,dir,row::Colon,col::Int) = AR(man,dir,1:size(man.peps,1),col)
AR(man,dir::Dir,row::UnitRange{Int64},col::Int) = [AR(man,dir,r,col) for r in row];
AR(man,dir,row::Int,col::Colon) = AR(man,dir,row,1:size(man.peps,2))
AR(man,dir::Dir,row::Int,col::UnitRange{Int64}) = [AR(man,dir,row,c) for c in col];


function AL(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AL[tc]
    else
        return man.infenvm.boundaries[dir].AL[tr,tc]
    end
end
function AL(man::FinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AL[tc]
end
function AL(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AL[tr,tc]
end

function AR(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AR[tc]
    else
        return man.infenvm.boundaries[dir].AR[tr,tc]
    end
end
function AR(man::FinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AR[tc]
end
function AR(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AR[tr,tc]
end

function AC(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AC[tc]
    else
        return man.infenvm.boundaries[dir].AC[tr,tc]
    end
end
function AC(man::FinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AC[tc]
end
function AC(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AC[tr,tc]
end

function CR(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=0 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].CR[tc]
    else
        return man.infenvm.boundaries[dir].CR[tr,tc]
    end
end
function CR(man::FinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].CR[tc]
end
function CR(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].CR[tr,tc]
end


#returns the relevant corner around peps tensor (row,col)
function corner(man,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.corners[dir][tr,tc]
end

function fp0LR(man,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp0[dir][tr,tc]
end
function fp1LR(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= size(man.fp1[dir],1) && tc >=1 && tc <= size(man.fp1[dir],2)
        return man.fp1[dir][tr,tc]
    else
        return man.infenvm.fp1[dir][tr,tc]
    end
end

function fp1LR(man::Union{FinEnvManager,InfEnvManager},dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp1[dir][tr,tc]
end

function fp0RL(man::Union{FinEnvManager,WinEnvManager},dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    @tensor tor[-1;-2]:=man.boundaries[left(dir)][tc].CR[end-tr+1][-1,1]*
    man.fp0[dir][tr,tc][1,2]*
    man.boundaries[right(dir)][end-tc+1].CR[tr-1][2,-2]
end
function fp0RL(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1;-2]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp0[dir][tr,tc][1,2]*
    man.boundaries[right(dir)].CR[end-tc+2,tr-1][2,-2]
end

function fp1RL(man::WinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if isin(man.peps,row,col)
        @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)][tc].CR[end-tr+1][-1,1]*
        man.fp1[dir][tr,tc][1,-2,-3,2]*
        man.boundaries[right(dir)][end-tc].CR[tr-1][2,-4]
        return tor
    else
        tman = rotate_north(man,dir);

        @tensor tor[-1 -2 -3;-4]:=CR(tman,West,tr,tc)[-1,1]*
        man.infenvm.fp1[dir][tr,tc][1,-2,-3,2]*
        CR(tman,East,tr-1,tc)[2,-4]
        return tor
    end
end
function fp1RL(man::FinEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)][tc].CR[end-tr+1][-1,1]*
    man.fp1[dir][tr,tc][1,-2,-3,2]*
    man.boundaries[right(dir)][end-tc].CR[tr-1][2,-4]
end
function fp1RL(man::InfEnvManager,dir::Dir,row::Int,col::Int)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp1[dir][tr,tc][1,-2,-3,2]*
    man.boundaries[right(dir)].CR[end-tc+1,tr-1][2,-4]
end
