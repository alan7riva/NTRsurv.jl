function credible_band( p::Float64, S::Matrix{Float64}, μ::Bool=true)
    if  !( 0 < p < 1)
        @error "ERROR: p is not between zero and one."
    end 
    l,k = size(S)
    m = round( Int, l*p)
    for _ in 1:m
        b = zeros(Float64, l)
        for i in 1:k
            b[i] = maximum(S[:,i])-minimum(S[:,i])
        end
        b_max_index = sortperm(b)[end]
        s_out_index = findmax( abs.( S[:,b_max_index] .- median(S[:,b_max_index]) ) )[2]
        S = S[1:end .!= s_out_index, 1:end  ]
    end
    band_u = [ maximum(S[:,i]) for i in 1:k]
    band_d = [ minimum(S[:,i]) for i in 1:k]
    band_m = zeros(k)
    if μ
        band_m = mean(S,dims=1)[1:end]
    else
        band_m = median(S,dims=1)[1:end]
    end
    return band_d, band_m, band_u
end

function prior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, α::Float64, baseline::Baseline, μ::Bool=true)
    S = sample_prior_survival(l,t,α,baseline)
    return credible_band( p, S, μ)
end

function posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, model::NeutralToTheRightModel, μ::Bool=true)
    S = sample_posterior_survival(l,t,model)
    return credible_band( p, S, μ)
end

function posterior_credible_band( p::Float64, l::Int64, t::Vector{Float64}, z_new::Vector{Float64}, model::CoxNeutralToTheRightModel, μ::Bool=true)
    S = sample_posterior_survival(l,t,z_new,model)
    return credible_band( p, S, μ)
end

function posterior_credible_band( p::Float64, t::Vector{Float64}, z_new::Vector{Float64}, model::CoxNeutralToTheRightFullyBayesianModel, μ::Bool=true)
    S = sample_posterior_survival(t,z_new,model)
    return credible_band( p, S, μ)
end