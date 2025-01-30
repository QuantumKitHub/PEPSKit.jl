include("simpleupdate.jl")

# continue SU with FU
peps = peps_
dts = [2e-2, 1e-2, 5e-3]
maxiter = 2000
colmove_alg = SequentialCTMRG(; verbosity=0, maxiter=1, trscheme=trscheme_envs)
CTMRGAlg = SequentialCTMRG
projector_alg = HalfInfiniteProjector
reconv_alg = CTMRGAlg(;
    tol=1e-6, maxiter=10, verbosity=2, trscheme=trscheme_envs, projector_alg
)
ctm_alg = CTMRGAlg(;
    tol=1e-10, maxiter=50, verbosity=2, trscheme=trscheme_envs, projector_alg
)
for dt in dts
    fu_alg = FullUpdate(;
        dt,
        maxiter,
        opt_alg=ALSTruncation(; trscheme=trscheme_peps),
        colmove_alg,
        reconv_alg,
    )
    result = fullupdate(peps, envs, ham, fu_alg, ctm_alg)
    global peps = result[1]
    global envs = result[2]
end
# measure physical quantities with CTMRG
meas = measure_heis(peps, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.66875; atol=1e-4)
@test isapprox(mean(meas["mag_norm"]), 0.3510; atol=2e-3)
