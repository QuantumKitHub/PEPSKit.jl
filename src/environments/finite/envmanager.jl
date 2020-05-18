struct FinEnvManager{P,A,T} <: EnvManager
    trscheme ::T

    planes :: Periodic{FinPlanes{P,A,T},1}
    corners :: Periodic{FinCorners{P,A,T},1}
end

function EnvManager(peps::FinPeps,trscheme)
    planes = Periodic(map(x->FinPlanes(peps,x,trscheme),[d for d in Dirs]))
    corners = Periodic(map(x->FinCorners(peps,x,trscheme),[d for d in Dirs]))

    return FinEnvManager(trscheme,planes,corners)
end
