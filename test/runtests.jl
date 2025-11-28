using SafeTestsets

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

@time begin
    if GROUP == "ALL" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin
            include("ctmrg/gaugefix.jl")
        end
        @time @safetestset "Unit cell" begin
            include("ctmrg/unitcell.jl")
        end
        @time @safetestset ":fixed CTMRG iteration scheme" begin
            include("ctmrg/fixed_iterscheme.jl")
        end
        @time @safetestset "SUWeight conversion" begin
            include("ctmrg/suweight.jl")
        end
        @time @safetestset "Flavors" begin
            include("ctmrg/flavors.jl")
        end
        @time @safetestset "Jacobian real linearity" begin
            include("ctmrg/jacobian_real_linear.jl")
        end
        @time @safetestset "Partition function" begin
            include("ctmrg/partition_function.jl")
        end
        @time @safetestset "PEPO" begin
            include("ctmrg/pepo.jl")
        end
        @time @safetestset "correlation length" begin
            include("ctmrg/correlation_length.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "BP"
        @time @safetestset "Unit cell bond matching" begin
            include("bp/unitcell.jl")
        end
        @time @safetestset "Expectation values" begin
            include("bp/expvals.jl")
        end
        @time @safetestset "Rotation of BPEnv" begin
            include("bp/rotation.jl")
        end
        @time @safetestset "Gauge-fixing iPEPS" begin
            include("bp/gaugefix.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "GRADIENTS"
        @time @safetestset "CTMRG gradients" begin
            include("gradients/ctmrg_gradients.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "BOUNDARYMPS"
        @time @safetestset "VUMPS" begin
            include("boundarymps/vumps.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "BONDENV"
        @time @safetestset "Iterative optimization after truncation" begin
            include("bondenv/bond_truncate.jl")
        end
        @time @safetestset "Gauge fixing" begin
            include("bondenv/benv_gaugefix.jl")
        end
        @time @safetestset "Full update bond environment" begin
            include("bondenv/benv_fu.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "TIMEEVOL"
        @time @safetestset "`timestep` function" begin
            include("timeevol/timestep.jl")
        end
        @time @safetestset "Cluster truncation with projectors" begin
            include("timeevol/cluster_projectors.jl")
        end
        @time @safetestset "Time evolution with site-dependent truncation" begin
            include("timeevol/sitedep_truncation.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "TOOLBOX"
        @time @safetestset "Density matrix from double-layer PEPO" begin
            include("toolbox/densitymatrices.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "UTILITY"
        @time @safetestset "LocalOperator" begin
            include("utility/localoperator.jl")
        end
        @time @safetestset "SVD wrapper" begin
            include("utility/svd_wrapper.jl")
        end
        @time @safetestset "Symmetrization" begin
            include("utility/symmetrization.jl")
        end
        @time @safetestset "Differentiable tmap" begin
            include("utility/diff_maps.jl")
        end
        @time @safetestset "Norm-preserving retractions" begin
            include("utility/retractions.jl")
        end
        @time @safetestset "Rotation of SUWeight" begin
            include("utility/suweight_rotation.jl")
        end
        @time @safetestset "Correlators" begin
            include("utility/correlator.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "EXAMPLES"
        @time @safetestset "Transverse field Ising model" begin
            include("examples/tf_ising.jl")
        end
        @time @safetestset "Transverse field Ising model at finite temperature" begin
            include("examples/tf_ising_finiteT.jl")
        end
        @time @safetestset "Heisenberg model" begin
            include("examples/heisenberg.jl")
        end
        @time @safetestset "J1-J2 model" begin
            include("examples/j1j2_model.jl")
        end
        @time @safetestset "J1-J2 model at finite temperature" begin
            include("examples/j1j2_finiteT.jl")
        end
        @time @safetestset "P-wave superconductor" begin
            include("examples/pwave.jl")
        end
        @time @safetestset "U1-symmetric Bose-Hubbard model" begin
            include("examples/bose_hubbard.jl")
        end
    end
end
