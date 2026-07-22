using Test, Random, Distributions, Survival, NTRsurv

function test_survival_curve(S::AbstractVector{<:Real}; atol=1e-12)
    @test all(isfinite.(S))
    @test S[1] ≈ 1.0 atol=atol
    @test all(diff(S) .<= atol)
    @test all((-atol .<= S) .& (S .<= 1.0 + atol))
end

function test_band_bounds(L, U; atol=1e-12)
    @test all(isfinite.(L))
    @test all(isfinite.(U))
    @test all(L .<= U .+ atol)
end

@testset "Constructor smoke tests" begin
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
    b = WeibullBaseline(k,λ)
    @test b isa Baseline
    b = ExponentialBaseline(1.0)
    @test b isa Baseline
    T = rand(Weibull(k,λ),90)
    δ = [ones(Int64,30);zeros(Int64,30);ones(Int64,30)]
    data = SurvivalData( T, δ )
    b = EmpiricalBayesBaseline(data)
    @test b isa Baseline
    b = EmpiricalBayesBaseline(data,exact=false)
    @test b isa Baseline
    model = NeutralToTheRightModel( 5.0, b, data)
    t =  collect(LinRange(0.0,maximum(T)+1,100)) # evaluation grid
    test_survival_curve(mean_posterior_survival(t, model))
    test_survival_curve(sample_posterior_survival(t, model))    
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,100,t,model) # posterior band computation
    test_band_bounds(NTR_band_d, NTR_band_u)
    test_survival_curve(NTR_band_m)
    Trep = [ T[1:30]; T[1:30]; T[1:30]]
    datarep = SurvivalData( Trep, δ )
    b = EmpiricalBayesBaseline(datarep,exact=false)
    @test b isa Baseline
    modelrep = NeutralToTheRightModel( 5.0, b, datarep)
    test_survival_curve(mean_posterior_survival(t, modelrep))
    test_survival_curve(sample_posterior_survival(t, modelrep))    
    NTR_band_d_rep, NTR_band_m_rep, NTR_band_u_rep = posterior_credible_band(0.05,100,t,modelrep) # posterior band computation
    test_band_bounds(NTR_band_d_rep, NTR_band_u_rep)
    test_survival_curve(NTR_band_m_rep)
    Z = [[randn()] for _ in 1:90]
    datareg = RegressionSurvivalData(T, δ, Z)
    cox_model = CoxNeutralToTheRightModel([0.5], 5.0, ExponentialBaseline(1.0), datareg)
    test_survival_curve(mean_posterior_survival(t, [0.2], cox_model))
    test_survival_curve(sample_posterior_survival(t, [0.2], cox_model))
    NTR_band_cox_d, NTR_band_cox_m, NTR_band_cox_u = posterior_credible_band(0.05,100,t,[0.2],cox_model) # posterior band computation
    test_band_bounds(NTR_band_cox_d, NTR_band_cox_u)
    test_survival_curve(NTR_band_cox_m)
    dataregrep = RegressionSurvivalData(Trep, δ, Z)
    cox_modelrep = CoxNeutralToTheRightModel([0.5], 5.0, ExponentialBaseline(1.0), dataregrep)
    test_survival_curve(mean_posterior_survival(t, [0.2], cox_modelrep))
    NTR_band_cox_d_rep, NTR_band_cox_m_rep, NTR_band_cox_u_rep = posterior_credible_band(0.05,100,t,[0.2],cox_modelrep) # posterior band computation
    test_band_bounds(NTR_band_cox_d_rep, NTR_band_cox_u_rep)
    test_survival_curve(NTR_band_cox_m_rep)
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
    Z = [[randn()] for _ in 1:90]
    data = SurvivalData(T,δ)
    b =  EmpiricalBayesBaseline(data)
    model = NeutralToTheRightModel( 5.0, b, data)
    t =  collect(LinRange(0.0,maximum(T)+1,300)) # evaluation grid
    test_survival_curve(mean_posterior_survival(t,model))
    test_survival_curve(sample_posterior_survival(t, model))    
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,3000,t,model) # posterior band computation
    test_band_bounds(NTR_band_d, NTR_band_u)
    test_survival_curve(NTR_band_m)
    datareg = RegressionSurvivalData(T, δ, Z)
    cox_model = CoxNeutralToTheRightModel([0.5], 5.0, ExponentialBaseline(1.0), datareg)
    test_survival_curve(mean_posterior_survival(t, [0.2], cox_model))
    test_survival_curve(sample_posterior_survival(t, [0.2], cox_model))
    NTR_band_cox_d_0, NTR_band_cox_m_0, NTR_band_cox_u_0 = posterior_credible_band(0.05,100,t,[0.0],cox_model)
    NTR_band_cox_d_1, NTR_band_cox_m_1, NTR_band_cox_u_1 = posterior_credible_band(0.05,100,t,[1.0],cox_model)
    NTR_band_cox_d_2, NTR_band_cox_m_2, NTR_band_cox_u_2 = posterior_credible_band(0.05,100,t,[-1.0],cox_model)
    test_band_bounds(NTR_band_cox_d_0, NTR_band_cox_u_0)
    test_band_bounds(NTR_band_cox_d_1, NTR_band_cox_u_1)
    test_band_bounds(NTR_band_cox_d_2, NTR_band_cox_u_2)
    test_survival_curve(NTR_band_cox_m_0)
    test_survival_curve(NTR_band_cox_m_1)
    test_survival_curve(NTR_band_cox_m_2)
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
    Z = [[randn()] for _ in 1:90]
    data = SurvivalData(T,δ)
    @test all( (data.nᵉ .+  data.nᶜ) .== n_reps )
    b =  EmpiricalBayesBaseline(data)
    model = NeutralToTheRightModel( 5.0, b, data)
    t =  collect(LinRange(0.0,maximum(T)+1,300)) # evaluation grid
    test_survival_curve(mean_posterior_survival(t,model))
    test_survival_curve(sample_posterior_survival(t, model))    
    NTR_band_d, NTR_band_m, NTR_band_u = posterior_credible_band(0.05,3000,t,model) # posterior band computation
    test_band_bounds(NTR_band_d, NTR_band_u)
    test_survival_curve(NTR_band_m)
    datareg = RegressionSurvivalData(T, δ, Z)
    cox_model = CoxNeutralToTheRightModel([0.5], 5.0, ExponentialBaseline(1.0), datareg)
    test_survival_curve(mean_posterior_survival(t, [0.2], cox_model))
    test_survival_curve(sample_posterior_survival(t, [0.2], cox_model))
    NTR_band_cox_d_0, NTR_band_cox_m_0, NTR_band_cox_u_0 = posterior_credible_band(0.05,100,t,[0.0],cox_model)
    NTR_band_cox_d_1, NTR_band_cox_m_1, NTR_band_cox_u_1 = posterior_credible_band(0.05,100,t,[1.0],cox_model)
    NTR_band_cox_d_2, NTR_band_cox_m_2, NTR_band_cox_u_2 = posterior_credible_band(0.05,100,t,[-1.0],cox_model)
    test_band_bounds(NTR_band_cox_d_0, NTR_band_cox_u_0)
    test_band_bounds(NTR_band_cox_d_1, NTR_band_cox_u_1)
    test_band_bounds(NTR_band_cox_d_2, NTR_band_cox_u_2)
    test_survival_curve(NTR_band_cox_m_0)
    test_survival_curve(NTR_band_cox_m_1)
    test_survival_curve(NTR_band_cox_m_2)
end

@testset "Weibull simulation study for NTR models" begin
    Random.seed!(1234)
    λ = 0.7 # true Weibull scale parameter
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
    test_survival_curve(Sm) # Test survival curve
    @test maximum( abs.(S0.-Sm) ) < 0.01 # consistency test with threshold
    _, SmMC, _ = posterior_credible_band(0.05,3000,t,model) # NTR posterior mean with Monte-Carlo computation
    test_survival_curve(SmMC) # Test survival curve
    @test maximum( abs.(S0.-SmMC) ) < 0.01 # consistency test with threshold
    @test maximum(abs.(Sm .- SmMC)) < 0.001 # Monte-Carlo and analytic implementations test
    _, Smed, _ = posterior_credible_band(0.05,3000,t,model,false) # NTR posterior median with Monte-Carlo computation
    test_survival_curve(Smed) # Test survival curve
    @test maximum( abs.(S0.-Smed) ) < 0.01 # consistency test with threshold
    # BvM tests for the NTR model without covariates
    Sdraws = sample_posterior_survival(3000,t,model) # NTR posterior draws computation
    km = fit(Survival.KaplanMeier, T, δ) # Kaplan-Meier fit with Survival.jl
    v = [0.8, 0.5, 0.2] # quantile levels
    km_i = [ findmin( abs.( km.survival .- q))[2] for q in v] # indexes for quantile levels
    t_i = [km.events.time[i] for i in km_i ] # times for quantile levels
    i_d = [ findmin( abs.( t .- ti))[2] for ti in t_i ] # indexes for draws at quantile levels 
    @test all([ abs(mean(Sdraws[:,i_d[i]]) - km.survival[ km_i[i] ]) < 0.01 for i in 1:length(v) ]) # BvM test of KM centerings
    R1_data = model.data.R₁[1:end-1] # risk-set counts from survival data struct
    ne_data = model.data.nᵉ # exact event counts from survival data struct
    U_event = n .* cumsum( ne_data ./ R1_data.^2 ) # observed Brownian time-change at observed times
    U_i = [
        begin
            j = searchsortedlast(model.data.T, t_i[i])
            j == 0 ? 0.0 : U_event[j]
        end
        for i in 1:length(v)
    ] # observed Brownian time-change at selected quantile times
    vars = [ var(Sdraws[:,i]) for i in i_d ] # posterior variances at selected quantile times
    var_asyms = [
        km.survival[km_i[i]]^2 * U_i[i] / n
        for i in 1:length(v)
    ] # BvM asymptotic variances with observed time-change
    @test all( [ abs(vars[i] - var_asyms[i]) < 0.00005 for i in 1:length(v)]) # BvM test for asymptotic variance 
    # Cox regression Weibull simulation study
    b = 0.3 # Cox regression coefficient
    Dz = MixtureModel(
        Normal[Normal(-2.5, 1.2), Normal(2.5, 1.2), Normal(0.0, 1.0)],
        [1/3, 1/3, 1/3]
    ) # covariate distribution
    Z = [[rand(Dz)] for _ in 1:n] # synthetic covariates
    Xregre = [ rand(Weibull(k, λ / exp(b * z[1] / k))) for z in Z ] # synthetic Cox-Weibull event times
    Cregre = rand(Exponential(3.75), n) # synthetic censoring variables
    Tregre = min.(Xregre,Cregre) # synthetic censored data generation
    δregre = 1*(Xregre .<= Cregre) # synthetic censoring indicators
    dataregre = RegressionSurvivalData(Tregre,δregre,Z) # regression survival data struct
    modelregre = CoxNeutralToTheRightModel([b], α, baseline, dataregre) # Cox NTR model struct
    z_0 = [0.0] # covariate for posterior computation
    z_1 = [2.5] # high risk covariate for posterior computation
    z_2 = [-2.5] # low risk covariate for posterior computation
    S0_regre_0 = exp.(-exp(b*z_0[1]) .* (t ./ λ).^k) # true survival for z_0
    S0_regre_1 = exp.(-exp(b*z_1[1]) .* (t ./ λ).^k) # true survival for z_1
    S0_regre_2 = exp.(-exp(b*z_2[1]) .* (t ./ λ).^k) # true survival for z_2
    Sm_regre_0 = mean_posterior_survival(t,z_0,modelregre) # Cox NTR posterior mean for z_0
    Sm_regre_1 = mean_posterior_survival(t,z_1,modelregre) # Cox NTR posterior mean for z_1
    Sm_regre_2 = mean_posterior_survival(t,z_2,modelregre) # Cox NTR posterior mean for z_2
    test_survival_curve(Sm_regre_0) # Test survival curve
    test_survival_curve(Sm_regre_1) # Test survival curve
    test_survival_curve(Sm_regre_2) # Test survival curve
    @test maximum( abs.(S0_regre_0.-Sm_regre_0) ) < 0.05 # consistency test with threshold
    @test maximum( abs.(S0_regre_1.-Sm_regre_1) ) < 0.05 # consistency test with threshold
    @test maximum( abs.(S0_regre_2.-Sm_regre_2) ) < 0.05 # consistency test with threshold
    _, Sm_regre_MC_0, _ = posterior_credible_band(0.05,3000,t,z_0,modelregre) # Cox NTR posterior mean with Monte-Carlo computation
    _, Sm_regre_MC_1, _ = posterior_credible_band(0.05,3000,t,z_1,modelregre) # Cox NTR posterior mean with Monte-Carlo computation
    _, Sm_regre_MC_2, _ = posterior_credible_band(0.05,3000,t,z_2,modelregre) # Cox NTR posterior mean with Monte-Carlo computation
    test_survival_curve(Sm_regre_MC_0) # Test survival curve
    test_survival_curve(Sm_regre_MC_1) # Test survival curve
    test_survival_curve(Sm_regre_MC_2) # Test survival curve
    @test maximum(abs.(Sm_regre_0 .- Sm_regre_MC_0)) < 0.01 # Monte-Carlo and analytic implementations test
    @test maximum(abs.(Sm_regre_1 .- Sm_regre_MC_1)) < 0.01 # Monte-Carlo and analytic implementations test
    @test maximum(abs.(Sm_regre_2 .- Sm_regre_MC_2)) < 0.01 # Monte-Carlo and analytic implementations test
    @test maximum( abs.(S0_regre_0.-Sm_regre_MC_0) ) < 0.055 # consistency test with threshold
    @test maximum( abs.(S0_regre_1.-Sm_regre_MC_1) ) < 0.055 # consistency test with threshold
    @test maximum( abs.(S0_regre_2.-Sm_regre_MC_2) ) < 0.055 # consistency test with threshold
    _, Smed_regre_0, _ = posterior_credible_band(0.05,3000,t,z_0,modelregre,false) # Cox NTR posterior median with Monte-Carlo computation
    _, Smed_regre_1, _ = posterior_credible_band(0.05,3000,t,z_1,modelregre,false) # Cox NTR posterior median with Monte-Carlo computation
    _, Smed_regre_2, _ = posterior_credible_band(0.05,3000,t,z_2,modelregre,false) # Cox NTR posterior median with Monte-Carlo computation
    test_survival_curve(Smed_regre_0) # Test survival curve
    test_survival_curve(Smed_regre_1) # Test survival curve
    test_survival_curve(Smed_regre_2) # Test survival curve
    @test maximum( abs.(S0_regre_0.-Smed_regre_0) ) < 0.055 # consistency test with threshold
    @test maximum( abs.(S0_regre_1.-Smed_regre_1) ) < 0.055 # consistency test with threshold
    @test maximum( abs.(S0_regre_2.-Smed_regre_2) ) < 0.055 # consistency test with threshold
    @test all(Sm_regre_1 .<= Sm_regre_0 .+ 1e-12) # Cox monotonicity check
    @test all(Sm_regre_0 .<= Sm_regre_2 .+ 1e-12) # Cox monotonicity check
    # BvM tests for Cox NTR model conditional on true regression coefficient
    R1_regre = modelregre.R₁[1:end-1] # Cox risk-set sums from Cox NTR model
    ne_regre = dataregre.nᵉ # exact event counts from regression survival data struct
    dK_regre = ne_regre ./ R1_regre # Breslow increments with true b
    K_event_regre = cumsum(dK_regre) # Breslow cumulative hazard at observed times
    U_event_regre = n .* cumsum( ne_regre ./ R1_regre.^2 ) # observed Brownian time-change
    Kbr_regre = [
        begin
            j = searchsortedlast(dataregre.T, ti)
            j == 0 ? 0.0 : K_event_regre[j]
        end
        for ti in t
    ] # Breslow cumulative hazard on evaluation grid
    U_regre = [
        begin
            j = searchsortedlast(dataregre.T, ti)
            j == 0 ? 0.0 : U_event_regre[j]
        end
        for ti in t
    ] # observed Brownian time-change on evaluation grid
    Sbr_regre_0 = exp.(-exp(b*z_0[1]) .* Kbr_regre) # Breslow-Cox estimator for z_0
    Sbr_regre_1 = exp.(-exp(b*z_1[1]) .* Kbr_regre) # Breslow-Cox estimator for z_1
    Sbr_regre_2 = exp.(-exp(b*z_2[1]) .* Kbr_regre) # Breslow-Cox estimator for z_2
    Sdraws_regre_0 = sample_posterior_survival(3000,t,z_0,modelregre) # Cox NTR posterior draws for z_0
    Sdraws_regre_1 = sample_posterior_survival(3000,t,z_1,modelregre) # Cox NTR posterior draws for z_1
    Sdraws_regre_2 = sample_posterior_survival(3000,t,z_2,modelregre) # Cox NTR posterior draws for z_2
    br_i_0 = [ findmin( abs.( Sbr_regre_0 .- q))[2] for q in v] # indexes for quantile levels
    br_i_1 = [ findmin( abs.( Sbr_regre_1 .- q))[2] for q in v] # indexes for quantile levels
    br_i_2 = [ findmin( abs.( Sbr_regre_2 .- q))[2] for q in v] # indexes for quantile levels
    @test all([ abs(mean(Sdraws_regre_0[:,br_i_0[i]]) - Sbr_regre_0[br_i_0[i]]) < 0.015 for i in 1:length(v) ]) # BvM test of Breslow-Cox centerings
    @test all([ abs(mean(Sdraws_regre_1[:,br_i_1[i]]) - Sbr_regre_1[br_i_1[i]]) < 0.015 for i in 1:length(v) ]) # BvM test of Breslow-Cox centerings
    @test all([ abs(mean(Sdraws_regre_2[:,br_i_2[i]]) - Sbr_regre_2[br_i_2[i]]) < 0.015 for i in 1:length(v) ]) # BvM test of Breslow-Cox centerings
    vars_regre_0 = [ var(Sdraws_regre_0[:,i]) for i in br_i_0 ] # posterior variances for z_0
    vars_regre_1 = [ var(Sdraws_regre_1[:,i]) for i in br_i_1 ] # posterior variances for z_1
    vars_regre_2 = [ var(Sdraws_regre_2[:,i]) for i in br_i_2 ] # posterior variances for z_2
    var_asyms_regre_0 = [
        Sbr_regre_0[br_i_0[i]]^2 * exp(b*z_0[1])^2 * U_regre[br_i_0[i]] / n
        for i in 1:length(v)
    ] # BvM asymptotic variances for z_0
    var_asyms_regre_1 = [
        Sbr_regre_1[br_i_1[i]]^2 * exp(b*z_1[1])^2 * U_regre[br_i_1[i]] / n
        for i in 1:length(v)
    ] # BvM asymptotic variances for z_1
    var_asyms_regre_2 = [
        Sbr_regre_2[br_i_2[i]]^2 * exp(b*z_2[1])^2 * U_regre[br_i_2[i]] / n
        for i in 1:length(v)
    ] # BvM asymptotic variances for z_2
    @test all( [ abs(vars_regre_0[i] - var_asyms_regre_0[i]) < 0.0001 for i in 1:length(v)]) # BvM test for asymptotic variance 
    @test all( [ abs(vars_regre_1[i] - var_asyms_regre_1[i]) < 0.0001 for i in 1:length(v)]) # BvM test for asymptotic variance 
    @test all( [ abs(vars_regre_2[i] - var_asyms_regre_2[i]) < 0.0001 for i in 1:length(v)]) # BvM test for asymptotic variance 
end