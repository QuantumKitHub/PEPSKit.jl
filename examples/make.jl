using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

using PlutoStaticHTML

const NOTEBOOK_DIR = joinpath(@__DIR__, "notebooks")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "docs", "src", "examples")

@info "Building notebooks in $NOTEBOOK_DIR"
oopts = OutputOptions(; append_build_context=true)
output_format = documenter_output
bopts = BuildOptions(
    NOTEBOOK_DIR; output_format, previous_dir=OUTPUT_DIR, max_concurrent_runs=1
)
build_notebooks(bopts, oopts)

@info "Copying markdown files"
for file in readdir(NOTEBOOK_DIR)
    _, ext = splitext(file)
    if ext == ".md"
        @debug "Copying $file"
        mv(joinpath(NOTEBOOK_DIR, file), joinpath(OUTPUT_DIR, file); force=true)
    end
end
