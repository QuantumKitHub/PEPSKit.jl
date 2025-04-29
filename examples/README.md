# Examples

This folder contains the examples and tutorials of this package. The files can be run as
scripts and are embedded into the docs using [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/).

## Building the documentation

The example files have to be built and updated manually. In order to trigger the file
generation, run:

``julia examples/make.jl`

By default, this will only generate files when the input file has not changed. This is
achieved by keeping a checksum of the `main.jl` file in each example in a `Cache.toml`.
Total recompilation can be achieved by deleting this file, or alternatively you can just
delete the entries for which you wish to generate new files.

## Contributing

Contributions are welcome! Please open an issue or a pull request if you have any questions
or suggestions. The examples should be placed in their own folder, where the `main.jl` file
serves as the entry point. Any other files will be copied over to the `docs/src/examples`
folder, so you can use this to include images or other files.
