ENV["GKSwstype"] = "100"
using Documenter, NTRsurv

makedocs(
    modules = [NTRsurv],
    checkdocs = :exports,
    sitename = "NTRsurv.jl",
    authors = "Alan Riva-Palacio",
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://alan7riva.github.io/NTRsurv.jl/stable/",
        edit_link = "main"
    ),
    pages = [
    "Home" => "index.md",
    "Getting started" => "getting_started.md",
    "Extended tutorials" => "tutorials.md",
    "API reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/alan7riva/NTRsurv.jl.git",
    devbranch = "main",
)