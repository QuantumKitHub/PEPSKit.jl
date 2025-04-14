# Examples and Tutorials

This directory contains the source code for the examples and tutorials of this package.
These are written using [Pluto.jl](https://plutojl.org/), and can be run as scripts or through Pluto itself.

## Building the documentation

As these examples can take substantial resources to build, they are not included in the automated documentation build.
To build the documentation, you can run the following command:

```bash
julia examples/make.jl
```

This should generate static HTML files in the `docs/examples` directory, which are then included in the documentation pages.
By default, examples that haven't changed since the last build are not re-run.

Note that Pluto uses its own package management system, which will use the latest registered version of this package.
This ensures the examples are always reproducible by themselves.
However, this also means that when registering a new version of this package, the next step is to re-run the examples to ensure that they are up-to-date, and build a new version of the documentation.

Alternatively, we can decide to manually keep a local environment for the examples by using `examples/Project.toml` and including the following block:

```julia
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # ensure latest version of PEPSKit is installed
    Pkg.dev(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
    
    # other packages
    using ...
end
```

For more information, see [Pluto.jl's documentation](https://plutojl.org/en/docs/packages-advanced/)
