### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ dbe8239a-f481-4681-95bc-ed464560a13b
using Random

# ╔═╡ 8c88c236-43da-497b-b083-dcb9ee7888a2
using TensorKit, PEPSKit

# ╔═╡ c115b454-1fa8-47b5-a510-b4be607dd8ba
using MPSKit: add_physical_charge

# ╔═╡ 118fb1e5-dce5-4eaf-9d85-7a2365409091
md"""
# Néel order in the U(1)-symmetric XXZ model

Here, we want to look at a special case of the Heisenberg model, where the ``x`` and ``y`` couplings are equal, called the XXZ model

```math
H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .
```

For appropriate ``\Delta``, the model enters an antiferromagnetic phase (Néel order) which we will force by adding staggered magnetic charges to ``H_0``. Furthermore, since the XXZ Hamiltonian obeys a ``U(1)`` symmetry, we will make use of that and work with ``U(1)``-symmetric PEPS and CTMRG environments. For simplicity, we will consider spin-``1/2`` operators.

But first, let's make this example deterministic and import the required packages:
"""

# ╔═╡ a43c4ba1-af3d-41a6-b3ea-e1c78f1fe461
Random.seed!(2928528935);

# ╔═╡ 2bec4073-864c-444b-a06d-aac1d7bdd9f7
md"""
## Constructing the model

Let us define the XXZ Hamiltonian with the parameters
"""

# ╔═╡ c633965b-82b7-4bb4-8a45-52a34ccc6a84
J, Delta, spin = 1.0, 1.0, 1//2;

# ╔═╡ b6f2d6f7-4be6-4067-ae4b-5f318531574b
md"""
with ``U(1)``-symmetric tensors
"""

# ╔═╡ 43857112-b6d5-4da8-9e5a-f55d48e5006a
symmetry = U1Irrep;

# ╔═╡ 08d95080-b23c-4dbd-bc47-6f7f5de53662
md"""
on a ``2 \times 2`` unit cell:
"""

# ╔═╡ 4d3f9e11-ec65-4187-a610-f38c5780506c
lattice = InfiniteSquare(2, 2);

# ╔═╡ eb2f129c-0169-4185-a878-4944a3d1fd4d
H₀ = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin);

# ╔═╡ a18b57e6-15dc-4147-a327-50c957ad96ca
md"""
This ensures that our PEPS ansatz can support the bipartite Néel order. As discussed above, we encode the Néel order directly in the ansatz by adding staggered auxiliary physical charges:
"""

# ╔═╡ f61f0b2f-da31-43c8-8d5e-6cb690b4d9ae
S_aux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
];

# ╔═╡ 7a257773-171b-4191-8011-f96f8754c571
H = add_physical_charge(H₀, S_aux);

# ╔═╡ 2dbf8595-a3b9-4836-a90f-335d7000cbd6
md"""
## Specifying the symmetric virtual spaces

Before we create an initial PEPS and CTM environment, we need to think about which symmetric spaces we need to construct. Since we want to exploit the global ``U(1)`` symmetry of the model, we will use TensorKit's `U1Space`s where we specify dimensions for each symmetry sector:
"""

# ╔═╡ 95aabe91-9be6-4fa1-8ee7-24c01f500f88
V_peps = U1Space(0 => 2, 1 => 1, -1 => 1);

# ╔═╡ f3528aa8-ef8d-4579-bdf5-95fedf9fdd08
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2);

# ╔═╡ 08213456-9a3f-44ed-99f3-11dff12cda9a
md"""
From the virtual spaces, we will need to construct a unit cell (a matrix) of spaces which will be supplied to the PEPS constructor. The same is true for the physical spaces, which we will just extract from the Hamiltonian `LocalOperator`:
"""

# ╔═╡ dc9c03ce-d620-4874-8e5b-e4d64758fc09
virtual_spaces = fill(V_peps, size(lattice)...);

# ╔═╡ 1088908b-fe26-4471-93ba-a8f4766582e3
physical_spaces = H.lattice;

# ╔═╡ b41e515b-8ba7-4cf6-8502-ed7cdd2a1e04
md"""
## Ground state search

From this point onwards it's business as usual: Create an initial PEPS and environment (using the symmetric spaces), specify the algorithmic parameters and optimize:
"""

# ╔═╡ 710db80f-468c-43a8-9899-cca2c95935e0
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace));

# ╔═╡ b4f46884-100a-4208-83d1-9ea11fd10e2f
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge);

# ╔═╡ 341f420e-ffbe-41b1-836b-75046c4ee62a
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2);

# ╔═╡ 7e3822b5-a1e7-4a2b-a6db-caa80004dfca
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces);

# ╔═╡ d75423c9-0c97-4711-97d1-5a812cd63c51
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);

# ╔═╡ d05a10e1-33b5-4c46-9f39-8604b12087fb
md"""
Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the optimization might take a while since precompilation of symmetric AD code takes longer and because symmetric tensors do create a bit of overhead (which does pay off at larger bond and environment dimensions):
"""

# ╔═╡ 642ab5a8-193a-11f0-01d7-3172b0c1a5c0
peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg);

# ╔═╡ d7d2e27d-01dc-4c3c-9ce9-6ac06581ec87
@show E

# ╔═╡ 8e7cc822-d193-4757-b54e-a21e2ba1009b
md"""
Note that for the specified parameters ``J=\Delta=1``, we simulated the same Hamiltonian as in the [Heisenberg example](). In that example, with a non-symmetric ``D=2`` PEPS simulation, we reached a ground-state energy of around ``E_\text{D=2} = -0.6625\dots``. Again comparing against
[Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate ``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
MPSKit = "bb1c41ca-d63c-52ed-829e-0820dda26502"
PEPSKit = "52969e89-939e-4361-9b68-9bc7cde4bdeb"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TensorKit = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"

[compat]
MPSKit = "~0.12.6"
PEPSKit = "~0.5.0"
TensorKit = "~0.14.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "c70e6e86b5621a8e0ee82a182ffe7c3e8712cf1e"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "4e25216b8fea1908a0ce0f5d87368587899f75be"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "df1746706459aa184ed3424b240a917b89d67b60"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.6.1"

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

    [deps.BlockArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"

[[deps.BlockTensorKit]]
deps = ["BlockArrays", "Compat", "LinearAlgebra", "Random", "Strided", "TensorKit", "TensorOperations", "TupleTools", "VectorInterface"]
git-tree-sha1 = "e0d455f2998aca33747def4f384c1aff5e3278de"
uuid = "5f87ffc2-9cf1-4a46-8172-465d160bd8cd"
version = "0.1.6"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "a975ae558af61a2a48720a6271661bf2621e0f4e"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.3"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "63a3903063d035260f0f6eab00f517471c5dc784"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.1.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "06d76c780d657729cf20821fb5832c6cc4dfd0b5"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.32"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

    [deps.InverseFunctions.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "PackageExtensionCompat", "Printf", "Random", "VectorInterface"]
git-tree-sha1 = "38477816f8db29956ea591feb3086d9edabf6f38"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.9.5"
weakdeps = ["ChainRulesCore"]

    [deps.KrylovKit.extensions]
    KrylovKitChainRulesCoreExt = "ChainRulesCore"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MPSKit]]
deps = ["Accessors", "BlockTensorKit", "Compat", "DocStringExtensions", "HalfIntegers", "KrylovKit", "LinearAlgebra", "LoggingExtras", "OhMyThreads", "OptimKit", "Printf", "Random", "RecipesBase", "TensorKit", "TensorKitManifolds", "TensorOperations", "VectorInterface"]
git-tree-sha1 = "588bef90e9672431efda1b686e8d477b621b2118"
uuid = "bb1c41ca-d63c-52ed-829e-0820dda26502"
version = "0.12.6"

[[deps.MPSKitModels]]
deps = ["LinearAlgebra", "MPSKit", "MacroTools", "PrecompileTools", "TensorKit", "TensorOperations", "TupleTools"]
git-tree-sha1 = "d658f13d6b1c08304344faede50bebcc4f574b6f"
uuid = "ca635005-6f8c-4cd1-b51d-8491250ef2ab"
version = "0.4.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "5f81bdb937fd857bac9548fa8ab9390a06864bb5"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.7.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+4"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OptimKit]]
deps = ["LinearAlgebra", "Printf", "ScopedValues", "VectorInterface"]
git-tree-sha1 = "b0163ac202bc03aeb15f0e55002e544e74965534"
uuid = "77e91f04-9b3b-57a6-a776-40b61faaebe0"
version = "0.4.0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PEPSKit]]
deps = ["Accessors", "ChainRulesCore", "Compat", "FiniteDifferences", "KrylovKit", "LinearAlgebra", "LoggingExtras", "MPSKit", "MPSKitModels", "OhMyThreads", "OptimKit", "Printf", "Random", "Statistics", "TensorKit", "TensorOperations", "VectorInterface", "Zygote"]
git-tree-sha1 = "12c42d39d9c67070d5783cbd7ebb54ab957c6a09"
uuid = "52969e89-939e-4361-9b68-9bc7cde4bdeb"
version = "0.5.0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RationalRoots]]
git-tree-sha1 = "e5f5db699187a4810fda9181b34250deeedafd81"
uuid = "308eb6b3-cc68-5ff3-9e97-c3c4da4fa681"
version = "0.2.1"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "48f038bfd83344065434089c2a79417f38715c41"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "4a1128f5237b5d0170d934a2eb4fa7a273639c9f"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.3.0"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "425158c52aa58d42593be6861befadf8b2541e9b"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.1"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "d155450e6dff2a8bc2fcb81dcb194bd98b0aeb46"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.2"

[[deps.TensorKit]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "Random", "SparseArrays", "Strided", "TensorKitSectors", "TensorOperations", "TupleTools", "VectorInterface"]
git-tree-sha1 = "4a09ea843d18b7256f710f1188414433dcf0b7db"
uuid = "07d1fe3e-3e46-537d-9eac-e9e13d0d4cec"
version = "0.14.5"
weakdeps = ["ChainRulesCore", "FiniteDifferences"]

    [deps.TensorKit.extensions]
    TensorKitChainRulesCoreExt = "ChainRulesCore"
    TensorKitFiniteDifferencesExt = "FiniteDifferences"

[[deps.TensorKitManifolds]]
deps = ["LinearAlgebra", "TensorKit"]
git-tree-sha1 = "94530ac4691795834e2efacb2d31d37571dad52f"
uuid = "11fa318c-39cb-4a83-b1ed-cdc7ba1e3684"
version = "0.7.2"

[[deps.TensorKitSectors]]
deps = ["HalfIntegers", "LinearAlgebra", "TensorOperations", "WignerSymbols"]
git-tree-sha1 = "fd15110964416b3ac3e7d1d212b0260e4629e536"
uuid = "13a9c161-d5da-41f0-bcbd-e1a08ae0647f"
version = "0.1.4"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PrecompileTools", "Preferences", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "c7458aad3d855e892e2cae208a793a5fceee2406"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.2.0"

    [deps.TensorOperations.extensions]
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationscuTENSORExt = ["cuTENSOR", "CUDA"]

    [deps.TensorOperations.weakdeps]
    Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9166406dedd38c111a6574e9814be83d267f8aec"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.5.0"

[[deps.WignerSymbols]]
deps = ["HalfIntegers", "LRUCache", "Primes", "RationalRoots"]
git-tree-sha1 = "960e5f708871c1d9a28a7f1dbcaf4e0ee34ee960"
uuid = "9f57e263-0b3d-5e2e-b1be-24f2bb48858b"
version = "2.0.0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "207d714f3514b0d564e3a08f9e9f753bf6566c2d"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.6"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╠═118fb1e5-dce5-4eaf-9d85-7a2365409091
# ╠═dbe8239a-f481-4681-95bc-ed464560a13b
# ╠═a43c4ba1-af3d-41a6-b3ea-e1c78f1fe461
# ╠═8c88c236-43da-497b-b083-dcb9ee7888a2
# ╠═c115b454-1fa8-47b5-a510-b4be607dd8ba
# ╠═2bec4073-864c-444b-a06d-aac1d7bdd9f7
# ╠═c633965b-82b7-4bb4-8a45-52a34ccc6a84
# ╠═b6f2d6f7-4be6-4067-ae4b-5f318531574b
# ╠═43857112-b6d5-4da8-9e5a-f55d48e5006a
# ╠═08d95080-b23c-4dbd-bc47-6f7f5de53662
# ╠═4d3f9e11-ec65-4187-a610-f38c5780506c
# ╠═eb2f129c-0169-4185-a878-4944a3d1fd4d
# ╠═a18b57e6-15dc-4147-a327-50c957ad96ca
# ╠═f61f0b2f-da31-43c8-8d5e-6cb690b4d9ae
# ╠═7a257773-171b-4191-8011-f96f8754c571
# ╠═2dbf8595-a3b9-4836-a90f-335d7000cbd6
# ╠═95aabe91-9be6-4fa1-8ee7-24c01f500f88
# ╠═f3528aa8-ef8d-4579-bdf5-95fedf9fdd08
# ╠═08213456-9a3f-44ed-99f3-11dff12cda9a
# ╠═dc9c03ce-d620-4874-8e5b-e4d64758fc09
# ╠═1088908b-fe26-4471-93ba-a8f4766582e3
# ╠═b41e515b-8ba7-4cf6-8502-ed7cdd2a1e04
# ╠═710db80f-468c-43a8-9899-cca2c95935e0
# ╠═b4f46884-100a-4208-83d1-9ea11fd10e2f
# ╠═341f420e-ffbe-41b1-836b-75046c4ee62a
# ╠═7e3822b5-a1e7-4a2b-a6db-caa80004dfca
# ╠═d75423c9-0c97-4711-97d1-5a812cd63c51
# ╠═d05a10e1-33b5-4c46-9f39-8604b12087fb
# ╠═642ab5a8-193a-11f0-01d7-3172b0c1a5c0
# ╠═d7d2e27d-01dc-4c3c-9ce9-6ac06581ec87
# ╠═8e7cc822-d193-4757-b54e-a21e2ba1009b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
