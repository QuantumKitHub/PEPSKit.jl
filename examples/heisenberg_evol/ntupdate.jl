include("simpleupdate.jl")

# continue SU with NTU
dts = [1e-2, 5e-3]
maxiter = 2000
for dt in dts
    alg = NTUpdate(;
        dt,
        maxiter,
        tol=1e-8,
        bondenv_alg=NTUEnvNN(),
        opt_alg=FullEnvTruncation(; trscheme=trscheme_peps),
        # opt_alg=ALSTruncation(; trscheme=trscheme_peps),
        ctm_alg=SequentialCTMRG(;
            tol=1e-7, verbosity=2, maxiter=30, trscheme=trscheme_envs
        ),
    )
    result = ntupdate(peps, envs, ham, alg, ctm_alg; bipartite=true)
    global peps = result[1]
    global envs = result[2]
end

# measure physical quantities with CTMRG
peps_ = InfinitePEPS(peps)
meas = measure_heis(peps_, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.66884; atol=1e-3)
@test isapprox(mean(meas["mag_norm"]), 0.34; atol=1e-2)
