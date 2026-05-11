using Test, Random, Distributions, Survival, NTRsurv

@testset "SurvivalData, Baseline and NeutralToTheRightModel constructor smoke tests" begin
    Random.seed!(1234)
    k = 2.0
    λ = 3.0
    κ(t) = (t / λ)^k
    dκ(t) = (k / λ) * (t / λ)^(k - 1)
    κinv(t) = λ*t^(1/k)
    b = Baseline(κ)
    @test b isa Baseline
    b = Baseline(κ,dκ)
    @test b isa Baseline
    b = Baseline(κ,dκ,κinv)
    @test b isa Baseline
    b = WeibullBaseline(λ,k)
    @test b isa Baseline
    b = ExponentialBaseline(1.0)
    @test b isa Baseline
    T = rand(Weibull(k,λ),90)
    δ = ones(Int64,90) 
    data = SurvivalData( T, δ )
    b = EmpiricalBayesBaseline(data)
    @test b isa Baseline
    b = EmpiricalBayesBaseline(data,exact=false)
    @test b isa Baseline
    model = NeutralToTheRightModel( 5.0, b, data)
    t =  collect(LinRange(0.0,maximum(T)+1,100)) # evaluation grid
    @test mean_posterior_survival(t,model) isa AbstractVector{<:Real}  # posterior mean survival computation
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,100,t,model) # posterior band computation
    @test NTR_band_d isa AbstractVector{<:Real} 
    @test NTR_band_m isa AbstractVector{<:Real} 
    @test NTR_band_u isa AbstractVector{<:Real} 
    @test sample_posterior_survival( t, model) isa AbstractVector{<:Real}  # posterior draw computation
    datarep = SurvivalData( [ T[1:30]; T[1:30]; T[1:30]], δ )
    b = EmpiricalBayesBaseline(datarep,exact=false)
    @test b isa Baseline
    modelrep = NeutralToTheRightModel( 5.0, b, datarep)
    @test mean_posterior_survival(t,modelrep) isa AbstractVector{<:Real}  # posterior mean survival computation
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,100,t,modelrep) # posterior band computation
    @test NTR_band_d isa Vector{Float64} 
    @test NTR_band_m isa Vector{Float64} 
    @test NTR_band_u isa Vector{Float64} 
    @test sample_posterior_survival( t, modelrep) isa AbstractVector{<:Real}  # posterior draw computation
end

@testset "Prior sampler correct mean matching" begin
    Random.seed!(1234)
    k = 2.0
    λ = 3.0
    κ(t) = (t / λ)^k
    b = Baseline(κ)
    t =  collect(LinRange(0.0,10.0,300))
    St = exp.(-κ.(t))
    _, prior_band_m, _ = prior_credible_band(0.05,3000,t,10.0,b)
    @test maximum( abs.( St .- prior_band_m ) ) < 0.01
end

@testset "NTR model without repetitions in data" begin
    Random.seed!(1234)
    T = Float64[
        310, 361, 654, 728,  61,  81, 520, 473, 107, 122, 965, 731, 153, 433, 146,  95, 765,
        735,   5, 687, 345, 444,  60, 208, 821, 305, 226, 426, 705, 363, 167, 641, 740, 245,
        588, 166, 559, 450, 529, 351, 205, 524, 199, 550, 551, 543, 293, 517, 511, 371, 201,
         62, 356, 340, 315, 182, 364, 376, 384, 268, 266, 194, 348, 382, 296, 186, 145, 269,
        350, 272, 292, 332, 285, 243, 276,  79, 240, 202, 235, 224, 239, 173, 252,  92, 192,
        211, 175, 203, 105, 177,
    ]
    δ = Int64[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1,
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ]
    data = SurvivalData(T,δ)
    b =  EmpiricalBayesBaseline(data)
    model = NeutralToTheRightModel( 5.0, b, data)
    function test_survival_curve(S::Vector{Float64})
        @test !any(isnan.(S))
        @test !any(isinf.(S))
        @test all(diff(S) .<= 0)
        @test all(0 .<= S .<= 1)
        @test S[1] ≈ 1 atol=1e-8
    end
    t =  collect(LinRange(0.0,maximum(T)+1,300)) # evaluation grid
    Sm = mean_posterior_survival(t,model) # posterior mean survival computation
    test_survival_curve(Sm)
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,3000,t,model) # posterior band computation
    @test all(NTR_band_d .<= NTR_band_m .<= NTR_band_u)
    test_survival_curve(NTR_band_d)
    test_survival_curve(NTR_band_m)
    test_survival_curve(NTR_band_u)
    Sdraw = sample_posterior_survival( t, model) # posterior draw computation
    test_survival_curve(Sdraw)
end

@testset "NTR model with repetitions in data" begin
    Random.seed!(1234)
    T = Float64[
        310, 361, 654, 728,  79,  79, 520, 473, 107, 122, 965, 731, 153, 433, 145,  79, 765,
        735,   5, 687, 345, 444,  79, 208, 735, 305, 79, 426, 705, 363, 208, 641, 740, 245,
        588, 166, 654, 450, 529, 351, 79, 524, 199, 550, 79, 543, 293, 511, 511, 371, 201,
         79, 266, 340, 315, 654, 364, 376, 384, 268, 266, 194, 348, 382, 296, 186, 145, 269,
        79, 79, 292, 332, 285, 243, 276,  79, 240, 202, 235, 224, 239, 173, 252,  79, 192,
        211, 166, 266, 105, 177,
    ]
    δ = Int64[
        1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1,
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ]
    n_reps = Int64[
        1, 12, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 
        1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1
    ]
    data = SurvivalData(T,δ)
    @test all( (data.nᵉ .+  data.nᶜ) .== n_reps )
    b =  EmpiricalBayesBaseline(data)
    model = NeutralToTheRightModel( 5.0, b, data)
    function test_survival_curve(S::Vector{Float64})
        @test !any(isnan.(S))
        @test !any(isinf.(S))
        @test all(diff(S) .<= 0)
        @test all(0 .<= S .<= 1)
        @test S[1] ≈ 1 atol=1e-8
    end
    t =  collect(LinRange(0.0,maximum(T)+1,300)) # evaluation grid
    Sm = mean_posterior_survival(t,model) # posterior mean survival computation
    test_survival_curve(Sm)
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,3000,t,model) # posterior band computation
    @test all(NTR_band_d .<= NTR_band_m .<= NTR_band_u)
    test_survival_curve(NTR_band_d)
    test_survival_curve(NTR_band_m)
    test_survival_curve(NTR_band_u)
    Sdraw = sample_posterior_survival( t, model) # posterior draw computation
    test_survival_curve(Sdraw)
end

@testset "Weibull simulation study for NTR model" begin
    Random.seed!(1234)
    λ= 0.7 # true Weibull rate parameter
    k = 2.5 # true Weibull shape parameter
    n = 9000 # sample size
    X = rand( Weibull(k,λ), n) # synthetic data generation
    C =  rand(Exponential(3.75), n) # synthetic censoring variables
    T = min.(X,C) # synthetic censored data generation
    δ = 1*(X .<= C)  # synthetic censoring indicators
    baseline = ExponentialBaseline(1.0) # standard exponential baseline 
    α = 5.0 # variance modulating hyper-parameter
    data = SurvivalData(T,δ) # survival data struct
    model = NeutralToTheRightModel( α, baseline, data) # NTR model struct
    t =  collect(LinRange(0.0,1.75,100)) # evaluation grid
    S0 = ccdf.( Weibull(k,λ), t) # true survival
    Sm = mean_posterior_survival(t,model) # NTR posterior mean computation
    @test maximum( abs.(S0.-Sm) ) < 0.01 # consistency test with threshold
    _, SmMC, _ = posterior_credible_band(0.05,3000,t,model) # NTR posterior mean with Monte-Carlo computation
    @test maximum( abs.(S0.-SmMC) ) < 0.01 # consistency test with threshold
    @test maximum(abs.(Sm .- SmMC)) < 0.001 # Monte-Carlo and analytic implementations test
    _, Smed, _ = posterior_credible_band(0.05,3000,t,model,false) # NTR posterior median with Monte-Carlo computation
    @test maximum( abs.(S0.-Smed) ) < 0.01 # consistency test with threshold
    Sdraws = sample_posterior_survival(3000,t,model) # NTR posterior draws computation
    km = fit(KaplanMeier, T, δ) # Kaplan-Meier fit with Survival.jl
    v = [0.8, 0.5, 0.2] # quantile levels
    km_i = [ findmin( abs.( km.survival .- q))[2] for q in v] # indexes for quantile levels
    t_i = [km.events.time[i] for i in km_i ] # times for quantile levels
    i_d = [ findmin( abs.( t .- ti))[2] for ti in t_i ] # indexes for draws at quantile levels 
    @test all([ abs(mean(Sdraws[:,i_d[i]]) - km.survival[ km_i[i] ]) < 0.01 for i in 1:length(v) ]) # BvN test of KM centerings
    vars = [ var(Sdraws[:,i]) for i in i_d ]
    var_asyms = [ccdf(Weibull(k,λ), t) *(1 - ccdf(Weibull(k,λ), t)) / n for t in t_i]
    @test all( [ abs(vars[i] - var_asyms[i])  < 0.00001 for i in 1:length(v)]) # BvM test for asymptotic varaince 
end

