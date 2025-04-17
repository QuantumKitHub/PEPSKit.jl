# Examples

This folder contains the examples and tutorials of this package. The files can be run as
scripts and are embedded into the docs using [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/).

## Building the documentation

The example files have to be built and updated manually. In order to trigger the file
generation, run:

``julia examples/make.jl`

By default, this will only generate files when the input file has not changed. This is
achieved by keeping a checksum of the `main.jl` file in each example in a `cache.toml`.
Total recompilation can be achieved by deleting this file, or alternatively you can just
delete the entries for which you wish to generate new files.

## Contributing

Contributions are welcome! Please open an issue or a pull request if you have any questions
or suggestions.
