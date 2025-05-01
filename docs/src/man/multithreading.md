# Multithreading

Before detailing the multithreading capabilities of PEPSKit, there are some general remarks to be made about parallelism in Julia.
In particular, it is important to know the interaction of Julia threads and BLAS threads, and that the BLAS thread behavior is inconsistent among different vendors.
Since these details have been explained many times elsewhere, we here want to point towards the [MPSKit docs](https://quantumkithub.github.io/MPSKit.jl/stable/man/parallelism/), which provide a good rundown of the threading behavior and what to be aware of.

PEPSKit's multithreading features are provided through [OhMyThreads.jl](https://juliafolds2.github.io/OhMyThreads.jl/stable/).
In addition, we also supply a differentiable parallel map, which parallelizes not only the forward pass but also the reverse pass of the map application, see [`PEPSKit.dtmap`](@ref).
The threading behaviour can be specified through a global `scheduler` that is interfaced through the [`set_scheduler!`](@ref) function:

```@docs
set_scheduler!
```

By default, the OhMyThreads machinery will be used to parallelize certain parts of the code, if Julia started with multiple threads.
Cases where PEPSKit can leverage parallel threads are:

- CTMRG steps using the `:simultaneous` scheme, where we parallelize over all unit cell coordinates and spatial directions
- The reverse pass of these parallelized CTMRG steps
- Evaluating expectation values of observables, where we parallelize over the terms contained in the `LocalOperator`
