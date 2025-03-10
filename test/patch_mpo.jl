using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

g = 3.044

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=3)
)

# initialize states
HOP = PEPSKit.patch_mpo_transverse_field_ising(; g=g)
Random.seed!(2928528935)
peps₀ = InfinitePEPS(2, χbond)
env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀, ctm_alg)

# find fixedpoint
peps, env, E, = fixedpoint(HOP, peps₀, env₀, opt_alg)
H = transverse_field_ising(InfiniteSquare(); g)
χ = [8 16 24 32 40 48 56 64 72 80]
E = zeros(length(χ))
EOP = zeros(length(χ))

for i in 1:length(χ)
    χenv = χ[i]
    ctm_alg = SimultaneousCTMRG(; trscheme=truncdim(χenv))
    env, = leading_boundary(env, peps, ctm_alg)
    E[i] = cost_function(peps, env, H)
    EOP[i] = cost_function(peps, env, HOP)
end

Random.seed!(2928528935)
χenv = 16
ctm_alg = SimultaneousCTMRG(; trscheme=truncdim(χenv))
test_no = 400
Et = zeros(test_no)
EtOP = zeros(test_no)
peps_coll = Vector{InfinitePEPS}(undef, test_no)
for i in 1:test_no
    peps = InfinitePEPS(2, χbond)
    env, = leading_boundary(CTMRGEnv(peps, ComplexSpace(χenv)), peps, ctm_alg)
    Et[i] = cost_function(peps, env, H)
    EtOP[i] = cost_function(peps, env, HOP)
    peps_coll[i] = peps
end

scatter((Et - EtOP); frame=:box)
plot!(; xlabel="Test number", ylabel="Relative energy difference", legend=false)

vm, im = findmax(abs.(Et - EtOP))
peps = peps_coll[im]

χ = [8 16 24 32 40 48 56 64 72 80]
E = zeros(length(χ))
EOP = zeros(length(χ))

for i in 1:length(χ)
    χenv = χ[i]
    ctm_alg = SimultaneousCTMRG(; trscheme=truncdim(χenv))
    env, = leading_boundary(env, peps, ctm_alg)
    E[i] = cost_function(peps, env, H)
    EOP[i] = cost_function(peps, env, HOP)
end

plot(
    χ[:],
    E[:];
    seriestype=:scatter,
    marker=(:circle, 4),
    label="2x1 patch with imaginary of -0.014",
    color=:red,
    frame=:box,
)
plot!(
    χ[:],
    EOP[:];
    seriestype=:scatter,
    marker=(:circle, 4),
    label="2x2 patch with imaginary -6.50e-6",
    color=:blue,
    frame=:box,
)
plot!(; xlabel="χenv", ylabel="Energy")
