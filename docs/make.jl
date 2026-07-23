using Documenter, NTRsurv

makedocs(
    modules = [NTRsurv],
    checkdocs = :exports,
    sitename = "NTRsurv.jl",
    authors = "Alan Riva-Palacio",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://alan7riva.github.io/NTRsurv.jl/stable/",
        edit_link = "main"
    ),
    pages = [
        "Home" => "home.md",
        #"Getting Started" => "getting_started.md",
        #"Event Times" => "events.md",
        #"Kaplan-Meier" => "km.md",
        #"Nelson-Aalen" => "na.md",
        #"Cox" => "cox.md",
    ],
)