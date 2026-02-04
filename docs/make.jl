using Documenter, NTRsurv

makedocs(sitename="NTRsurv.jl")


makedocs(
    modules = [NTRsurv],
    checkdocs = :exports,
    sitename = "NTRsurv.jl",
    authors = "Alan Riva-Palacio",
    pages = [
        "Home" => "index.md",
        #"Getting Started" => "getting_started.md",
        #"Event Times" => "events.md",
        #"Kaplan-Meier" => "km.md",
        #"Nelson-Aalen" => "na.md",
        #"Cox" => "cox.md",
    ],
)