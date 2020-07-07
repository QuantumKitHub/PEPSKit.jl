#=
    Some utility - eyecandy getters

    the way things are stored is not very nice to actually work with in algorithms
    hence some helpers to facilitate everything
=#
struct Transformed_Propview{E,T<:Union{InfEnvManager,FinEnvManager,WinEnvManager}} <: AbstractArray{E,3}
    envm::T
    fun::Function
end

#getindex of currently stored peps - environment
function Base.getindex(d::Transformed_Propview{E,T},dir::Int,row::Int,col::Int) where{E,T}
    d.fun(d.envm,dir,row,col)::E
end

Base.setindex!(d::Transformed_Propview,args...) = throw(ArgumentError("not supported"))
Base.size(d::Transformed_Propview) = (length(Dirs),size(d.envm.peps,1),size(d.envm.peps,2))

function Base.getproperty(man::Union{InfEnvManager,FinEnvManager,WinEnvManager},prop::Symbol)
    if prop == :AL ||
        prop == :AC ||
        prop == :AR ||
        prop == :fp1LR ||
        prop == :fp1RL

        elt = eltype(man.fp1[1])
        return Transformed_Propview{elt,typeof(man)}(man,getfield(PEPSKit,prop))
    elseif prop == :CR ||
        prop == :corner ||
        prop == :fp0LR ||
        prop == :fp0RL

        elt = eltype(man.corners[1]);
        return Transformed_Propview{elt,typeof(man)}(man,getfield(PEPSKit,prop))
    else
        return getfield(man,prop)
    end
end

function AL(man::WinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AL[tc]
    else
        return man.infenvm.boundaries[dir].AL[tr,tc]
    end
end
function AL(man::FinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AL[tc]
end
function AL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AL[tr,tc]
end

function AR(man::WinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AR[tc]
    else
        return man.infenvm.boundaries[dir].AR[tr,tc]
    end
end
function AR(man::FinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AR[tc]
end
function AR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AR[tr,tc]
end

function AC(man::WinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=1 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].AC[tc]
    else
        return man.infenvm.boundaries[dir].AC[tr,tc]
    end
end
function AC(man::FinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].AC[tc]
end
function AC(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AC[tr,tc]
end

function CR(man::WinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    if tr >= 1 && tr <= length(man.boundaries[dir]) && tc >=0 && tc <= length(man.boundaries[dir][tr])
        return man.boundaries[dir][tr].CR[tc]
    else
        return man.infenvm.boundaries[dir].CR[tr,tc]
    end
end
function CR(man::FinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir][tr].CR[tc]
end
function CR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].CR[tr,tc]
end


#returns the relevant corner around peps tensor (row,col)
function corner(man,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.corners[dir][tr,tc]
end

function fp0LR(man,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp0[dir][tr,tc]
end
function fp1LR(man::WinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    if tr >= 1 && tr <= size(man.fp1[dir],1) && tc >=1 && tc <= size(man.fp1[dir],2)
        return man.fp1[dir][tr,tc]
    else
        return man.infenvm.fp1[dir][tr,tc]
    end
end

function fp1LR(man::Union{FinEnvManager,InfEnvManager},dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp1[dir][tr,tc]
end

function fp0RL(man::Union{FinEnvManager,WinEnvManager},dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    @tensor tor[-1;-2]:=man.boundaries[left(dir)][tc].CR[end-tr+1][-1,1]*
    man.fp0[dir][tr,tc][1,2]*
    man.boundaries[right(dir)][end-tc+1].CR[tr-1][2,-2]
end
function fp0RL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1;-2]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp0[dir][tr,tc][1,2]*
    man.boundaries[right(dir)].CR[end-tc+2,tr-1][2,-2]
end

function fp1RL(man::WinEnvManager,dir,row,col)
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
function fp1RL(man::FinEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)][tc].CR[end-tr+1][-1,1]*
    man.fp1[dir][tr,tc][1,-2,-3,2]*
    man.boundaries[right(dir)][end-tc].CR[tr-1][2,-4]
end
function fp1RL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp1[dir][tr,tc][1,-2,-3,2]*
    man.boundaries[right(dir)].CR[end-tc+1,tr-1][2,-4]
end
