# Based on the design of GPUArrays.jl

"""
    TestSuite

Suite of tests that may be used for all packages inheriting from MatrixAlgebraKit.

Every test file is included into its own submodule, so that the parameters a file
defines at top level (`D`, `χ`, `unitcells`, ...) stay private to that file instead
of clobbering the identically named parameters of its neighbours. Each submodule
exports the test entry points it defines, which `TestSuite` re-exports in turn, so
that callers keep using `TestSuite.some_test(AT)`.

"""
module TestSuite

# Wrap everything in new modules to keep various tests from clobbering each others'
# variables

# BondEnv
# -------
module BondEnvCTM
    include("bondenv/benv_ctm.jl")
    export bondenv_ctm
end
using .BondEnvCTM

module BondEnvGaugeFix
    include("bondenv/benv_gaugefix.jl")
    export bondenv_gaugefix
end
using .BondEnvGaugeFix

module BondEnvTruncate
    include("bondenv/bond_truncate.jl")
    export bondenv_truncate
end
using .BondEnvTruncate

# BoundaryMPS
# -----------
module BoundaryMPSVUMPS
    include("boundarymps/vumps.jl")
    export boundary_mps_one_one_peps, boundary_mps_two_two_peps
    export boundary_mps_fermionic_peps, boundary_mps_pepo_runthrough
end
using .BoundaryMPSVUMPS

# BP
# --------------
module BPExpVals
    include("bp/expvals.jl")
    export bp_expvals
end
using .BPExpVals

module BPGaugeFix
    include("bp/gaugefix.jl")
    export bp_gaugefix_bp_vs_su
end
using .BPGaugeFix

module BPRotation
    include("bp/rotation.jl")
    export bp_rotations
end
using .BPRotation

module BPUnitCell
    include("bp/unitcell.jl")
    export bp_unitcell_random_cartesian_spaces, bp_unitcell_specific_u1_spaces
end
using .BPUnitCell

# Compress
# --------
module CompressLocal
    include("compress/local.jl")
    export compress_fermionic_twists, compress_cost_function
    export compress_virtual_space_matching
end
using .CompressLocal

# CTMRG
# -----
module CTMRGContractions
    include("ctmrg/contractions.jl")
    export ctmrg_contractions_random_cartesian_spaces, ctmrg_contractions_specific_u1_spaces
    export ctmrg_contractions_random_fermionic_spaces
end
using .CTMRGContractions

module CTMRGCorrelationLength
    include("ctmrg/correlation_length.jl")
    export ctmrg_correlation_length
end
using .CTMRGCorrelationLength

module CTMRGInitialization
    include("ctmrg/initialization.jl")
    export ctmrg_initialization_critical_ising, ctmrg_initialization_peps
end
using .CTMRGInitialization

module CTMRGJacobianRealLinear
    include("ctmrg/jacobian_real_linear.jl")
    export ctmrg_jacobian_real_linear
end
using .CTMRGJacobianRealLinear

module CTMRGFixedIterScheme
    include("ctmrg/fixed_iterscheme.jl")
    export ctmrg_fixed_iterscheme_asymmetric, ctmrg_fixed_iterscheme_c4v
    export ctmrg_fixed_iterscheme_divide_and_conquer
end
using .CTMRGFixedIterScheme

module CTMRGFlavors
    include("ctmrg/flavors.jl")
    export ctmrg_flavors_unitcells, ctmrg_flavors_fixedspace_truncation, ctmrg_flavors_c4v
end
using .CTMRGFlavors

module CTMRGGaugeFix
    include("ctmrg/gaugefix.jl")
    export ctmrg_gaugefix_asymmetric, ctmrg_gaugefix_c4v
end
using .CTMRGGaugeFix

module CTMRGPartitionFunction
    include("ctmrg/partition_function.jl")
    export ctmrg_partition_function, ctmrg_partition_function_spaces
end
using .CTMRGPartitionFunction

module CTMRGPEPO
    include("ctmrg/pepo.jl")
    export ctmrg_pepo_runthroughs, ctmrg_pepo_fixed_point
end
using .CTMRGPEPO

module CTMRGSUWeight
    include("ctmrg/suweight.jl")
    export ctmrg_suweight
end
using .CTMRGSUWeight

module CTMRGUnitCell
    include("ctmrg/unitcell.jl")
    export ctmrg_unitcell_random_cartesian_spaces, ctmrg_unitcell_specific_u1_spaces
end
using .CTMRGUnitCell

# Examples
# --------
module ExamplesBoseHubbard
    include("examples/bose_hubbard.jl")
    export examples_bose_hubbard
end
using .ExamplesBoseHubbard

module ExamplesHeisenberg
    include("examples/heisenberg.jl")
    export examples_heisenberg
end
using .ExamplesHeisenberg

module ExamplesJ1J2
    include("examples/j1j2_model.jl")
    export examples_j1j2
end
using .ExamplesJ1J2

module ExamplesPWave
    include("examples/pwave.jl")
    export examples_pwave
end
using .ExamplesPWave

module ExamplesTFIsing
    include("examples/tf_ising.jl")
    export examples_tf_ising
end
using .ExamplesTFIsing

# Gradients
# ---------
module GradientsC4vCTMRG
    include("gradients/c4v_ctmrg_gradients.jl")
    export gradients_c4v
end
using .GradientsC4vCTMRG

module GradientsCTMRG
    include("gradients/ctmrg_gradients.jl")
    export gradients_asymmetric, gradients_asymmetric_276
end
using .GradientsCTMRG

# Time evolution
# --------------
module TimeEvolClusterProjectors
    include("timeevol/cluster_projectors.jl")
    export timeevol_cluster_bond_truncation, timeevol_cluster_identity_gate
    export timeevol_cluster_hubbard
end
using .TimeEvolClusterProjectors

module TimeEvolSiteDepTruncation
    include("timeevol/sitedep_truncation.jl")
    export timeevol_sitedep_rotation, timeevol_sitedep_su
end
using .TimeEvolSiteDepTruncation

module TimeEvolJ1J2FiniteT
    include("timeevol/j1j2_finiteT.jl")
    export timeevol_j1j2_finiteT
end
using .TimeEvolJ1J2FiniteT

module TimeEvolTimestep
    include("timeevol/timestep.jl")
    export timeevol_timestep
end
using .TimeEvolTimestep

module TimeEvolTFIsingFiniteT
    include("timeevol/tf_ising_finiteT.jl")
    export timeevol_ising_finiteT
end
using .TimeEvolTFIsingFiniteT

# Toolbox
# -------
module ToolboxDensityMatrices
    include("toolbox/densitymatrices.jl")
    export toolbox_single_layer_densitymatrix, toolbox_double_layer_densitymatrix
    export toolbox_densitymatrix_too_many_layers, toolbox_densitymatrix_generic_fallback
end
using .ToolboxDensityMatrices

module ToolboxTensorProductTerms
    include("toolbox/tensorproduct_terms.jl")
    export toolbox_tensorproduct_ising, toolbox_tensorproduct_bookkeeping
end
using .ToolboxTensorProductTerms

module ToolboxMPOTerms
    include("toolbox/mpo_terms.jl")
    export toolbox_mpo_convention, toolbox_mpo_dispatch, toolbox_mpo_terms_dense
    export toolbox_mpo_heisenberg, toolbox_mpo_bookkeeping
end
using .ToolboxMPOTerms

# Utility
# -------
module UtilityCorrelator
    include("utility/correlator.jl")
    export utility_correlator_infinite_peps, utility_correlator_purified_ipepo
    export utility_correlator_single_layer_ipepo
end
using .UtilityCorrelator

module UtilityDiffMaps
    include("utility/diff_maps.jl")
    export utility_diff_maps
end
using .UtilityDiffMaps

module UtilityEighWrapper
    include("utility/eigh_wrapper.jl")
    export utility_eigh_wrapper
end
using .UtilityEighWrapper

module UtilitySVDWrapper
    include("utility/svd_wrapper.jl")
    export utility_svd_wrapper
end
using .UtilitySVDWrapper

module UtilityRetractions
    include("utility/retractions.jl")
    export utility_retractions
end
using .UtilityRetractions

module UtilitySymmetrization
    include("utility/symmetrization.jl")
    export utility_symmetrization_reflect_depth, utility_symmetrization_reflect_width
    export utility_symmetrization_rotate, utility_symmetrization_rotate_reflect
end
using .UtilitySymmetrization

end
