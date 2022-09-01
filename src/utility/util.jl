function sdiag_inv_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    if sectortype(S) == Trivial
        copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
    else
        for (k,b) in blocks(S)
            copyto!(blocks(toret)[k],LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2)));
        end
    end
    toret
end
function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap)
    toret = sdiag_inv_sqrt(S);
    toret,c̄ -> (ChainRulesCore.NoTangent(),-1/2*_elementwise_mult(c̄,toret'^3))
end
function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end

#rotl90 appeared to lose PeriodicArray'ness, which in turn caused zygote problems
Base.rotl90(a::PeriodicArray) = PeriodicArray(rotl90(a.data));
Base.rotr90(a::PeriodicArray) = PeriodicArray(rotr90(a.data));
function ChainRulesCore.rrule(::typeof(rotl90),a::AbstractMatrix)
    pr_a = ProjectTo(a);
    function pb(x)
        if !iszero(x)
            x = rotr90(pr_a(x));
        end

        (ZeroTangent(),x)
    end
    rotl90(a), pb
end
ChainRulesCore.ProjectTo(xs::T) where T<:PeriodicArray = ProjectTo{T}(;axes = axes(xs));
function (project::ChainRulesCore.ProjectTo{PeriodicArray{T,N}})(m::Array{<:Any}) where {T,N}
    PeriodicArray(reshape(m,project.axes)) # m can contain both T, but also ZeroTangent, or NoTangent.... => it cannot be a PeriodicArray{T,N}
end

function (project::ChainRulesCore.ProjectTo{PeriodicArray{T,N}})(m::ChainRulesCore.Tangent{Any,D}) where {T,N,D}
    PeriodicArray(reshape(m.data,project.axes))
end

structure(t) = codomain(t)←domain(t);

function _setindex(a::AbstractArray,v,args...)
    b::typeof(a) = copy(a);
    b[args...] = v
    b
end
function ChainRulesCore.rrule(::typeof(_setindex),a::AbstractArray,tv,args...) 
    t = _setindex(a,tv,args...);
    pr_a = ProjectTo(a);
    function toret(v)
        lol = copy(pr_a(v));
        lol[args...] = zero.(lol[args...]);
        (NoTangent(),lol,pr_a(v)[args...],fill(ZeroTangent(),length(args))...)
    end
    t,toret
end

macro diffset(ex)
    esc(parse_ex(ex));
end
parse_ex(ex) = ex
function parse_ex(ex::Expr)
    oppheads = (:(./=),:(.*=),:(.+=),:(.-=));
    opprep = (:(./),:(.*),:(.+),:(.-));
    if ex.head == :macrocall
        parse_ex(macroexpand(PEPSKit,ex))
    elseif ex.head in ( :(.=), :(=)) && length(ex.args)==2 && is_indexing(ex.args[1])
        lhs = ex.args[1];
        rhs = ex.args[2];

        vname = lhs.args[1];
        args = lhs.args[2:end];
        toret = quote
            $vname = _setindex($vname,$rhs,$(args...))
        end

        return toret
    elseif ex.head in oppheads &&  length(ex.args)==2 && is_indexing(ex.args[1])
        hit = findfirst(x->x==ex.head,oppheads);
        rep = opprep[hit];

        lhs = ex.args[1];
        rhs = ex.args[2];

        vname = lhs.args[1];
        args = lhs.args[2:end];
        
        toret = quote
            $vname = _setindex($vname,$(rep)($lhs,$rhs),$(args...))
        end

        return toret
    else
        return Expr(ex.head,parse_ex.(ex.args));
    end
end

is_indexing(ex) = false
is_indexing(ex::Expr) = ex.head == :ref
