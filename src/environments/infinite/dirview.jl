#=
    Some utility - eyecandy getters

    the way things are stored is not very nice to actually work with in algorithms
    hence some helpers to facilitate everything

    I did (unnecessarily) sacrifice type stability - add back in later!
=#
struct Transformed_Propview{T<:InfEnvManager} <: AbstractArray{Any,3}
    envm::T
    fun::Symbol
end

function Base.getindex(d::Transformed_Propview,peps::InfPEPS,dir::Int,row::Int,col::Int)
    if peps != d.envm.peps
        #using warn because this is still kinda experimental
        @warn "environment of different peps, recalculating"
        recalculate!(d.envm,peps)
    end
    @eval $(d.fun)($(d.envm),$(dir),$(row),$(col))
end

#getindex of currently stored peps - environment
function Base.getindex(d::Transformed_Propview,dir::Int,row::Int,col::Int)
    @eval $(d.fun)($(d.envm),$(dir),$(row),$(col))
end

Base.setindex!(d::Transformed_Propview,args...) = throw(ArgumentError("not supported"))
Base.size(d::Transformed_Propview) = (4,size(d.envm.peps,1),size(d.envm.peps,2))

function Base.getproperty(man::InfEnvManager,prop::Symbol)

    if  prop == :AL ||
        prop == :AC ||
        prop == :AR ||
        prop == :CR ||
        prop == :corner ||
        prop == :fp0LR ||
        prop == :fp0RL ||
        prop == :fp1LR ||
        prop == :fp1RL ||
        prop == :fp2LR ||
        prop == :fp2RL
        return Transformed_Propview(man,prop)
    else
        return getfield(man,prop)
    end
end

function AL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AL[tr,tc]
end
function AR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AR[tr,tc]
end
function AC(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].AC[tr,tc]
end
function CR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.boundaries[dir].CR[tr,tc]
end
#returns the relevant corner around peps tensor (row,col)
function corner(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.corners[dir][tr,tc]
end

function fp0LR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp0[dir][tr,tc]
end
function fp1LR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp1[dir][tr,tc]
end

function fp2LR(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);
    return man.fp2[dir][tr,tc]
end

function fp0RL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1;-2]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp0[dir][tr,tc][1,2]*
    man.boundaries[right(dir)].CR[end-tc+2,tr-1][2,-2]
end
function fp1RL(man::InfEnvManager,dir,row,col)
    (tr,tc)=rotate_north((row,col),size(man.peps),dir);

    @tensor tor[-1 -2 -3;-4]:=man.boundaries[left(dir)].CR[tc,end-tr+1][-1,1]*
    man.fp1[dir][tr,tc][1,-2,-3,2]*
    man.boundaries[right(dir)].CR[end-tc+1,tr-1][2,-4]
end
