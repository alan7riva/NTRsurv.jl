# In this file we include code for computation of fequentist confidence bands,
# which is used in some of the examples folder tutorials.
import LinearAlgebra: dot
# Auxiliary step-function evaluation of  right-continuous step function at a single point.
function step_eval(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    q::Real;
    left::Real = 1.0
)
    i = searchsortedlast(x, q)
    return i == 0 ? Float64(left) : Float64(y[i])
end
# Auxiliary step-function evaluation of  right-continuous step function at several points.
function step_eval(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    q::AbstractVector{<:Real};
    left::Real = 1.0
)
    return [step_eval(x, y, qi; left = left) for qi in q]
end
# Estimator for NTR Bernstein-von Mises theorem assymptotic Gaussian time scale U₀ 
function Uestim(t::Vector{Float64},data::SurvivalData)
    n = data.n
    return step_eval(data.T, n *  cumsum(  data.nᵉ./data.R₁[1:end-1].^2.0 ), t; left=0.0)
end
# Sampler of asymptotic Gaussian processes in NTR Bernstein-von Mises theorem
function BatU(t::Vector{Float64},U::Function)
    v = zeros(length(t))
    if t[1] == 0.0
        v[1] = 0.0
    else
        v[1] = rand(Normal(0,sqrt( U(t[1])) ))
    end
    for i in 2:length(t)
        v[i] = v[i-1] + rand(Normal(0,sqrt( U(t[i]) - U(t[i-1]) ) ))
    end
    return v
end
# Auxilliary function for `l` simulations of the asymptotic NTR Bernstein-von Mises theorem Gaussian distributions in an array `t`
# ordered in an `l×length(t)` matrix
function Sfreq_mat(l::Int64,n::Int64,t::Vector{Float64},Skm::Vector{Float64},U::Function)
    S_mat =  Matrix{eltype(t)}(undef, l, length(t))
    for i in 1:l
        S_mat[i,:] = Skm .* exp.( -BatU(t,U)/sqrt(n) )
    end
    return S_mat
end
# Breslow estimator for cumulative hazard in frequentist Cox models
function BreslowEstimCoxModel(model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    status = getfield.(model.mf.data.event, :status)
    n = length(status)
    M =  modelmatrix(model)
    η = M * c_freq
    risk = exp.(η)
    R = cumsum( risk[end:-1:1] )[end:-1:1]
    return cumsum( [ (δ[i]==1) ? 1.0./R[i] : 0.0 for i in 1:n] ) 
end
# Auxilliary function for numerical integration with trapezoidal rule
function trapz(x::AbstractVector, y::AbstractVector)
    return sum( 0.5 .* (y[1:end-1] .+ y[2:end]) .* diff(x) )
end
# Auxilliary function for numerical integration with trapezoidal rule with cumulative output
function cumtrapz(x::AbstractVector, y::AbstractVector)
    out = zeros(length(x))
    for j in 2:length(x)
        out[j] = out[j-1] + 0.5 * (y[j-1] + y[j]) * (x[j] - x[j-1])
    end
    return out
end
# Monte-Carlo computation for Cox regression asymptotic variance components, Fisher information 
# per observation `I_MC`, extra regression coefficient variance component `V_MC` and assymptotic 
# Gaussian time scale `U_MC`
function cox_vars_MC(M::Int,t::Vector{Float64},b₀::Float64,k::Float64,λ::Float64,Dz::Distribution,b::Float64 = b₀)
    MC_Z = rand(Dz, M)
    MC_Zv = [ z[1] for z in MC_Z ] 
    l = length(t)
    H = (t./λ).^k
    r₀ =  exp.( b₀ .* MC_Zv )
    logr = b .* MC_Zv  
    Z2 = MC_Zv .^ 2
    S0 = zeros(l)
    S1 = zeros(l)
    S2 = zeros(l)
    integrand = zeros(l)
    for j in 1:l
        w = exp.( logr .- r₀.*H[j] )
        S0[j] = mean( w )
        S1[j] = mean( MC_Zv.*w )
        S2[j] = mean( MC_Zv.^2.0 .* exp.( logr .- r₀.*H[j] ) )
        integrand[j] = S2[j] - S1[j]^2 / S0[j]
    end
    I_MC = trapz(H, integrand)
    U_MC = cumtrapz(H, 1.0 ./ S0)
    V_MC = cumtrapz(H, S1 ./ S0)
    return I_MC, V_MC, U_MC
end

# Estimators for observed Cox regression asymptotic variance components, Fisher information 
# per observation `I_obs`, extra regression coefficient variance component `V_obs` and assymptotic 
# Gaussian time scale `U_obs`
function cox_var_obs(t::Vector{Float64},c::Float64,D::RegressionSurvivalData)
    n = length(D.Z)
    η = [dot( c, v) for v in   D.Z]
    Zv = [ z[1] for z in D.Z ] 
    w = exp.(η)
    # Risk sums: R_i(c) = sum_{j: T_j >= T_i} exp(c z_j)
    S0obs = reverse(cumsum(reverse(w)))
    S1obs = reverse(cumsum(reverse(w .* Zv)))
    S2obs = reverse(cumsum(reverse(w .* Zv.^2.0)))
    # Observed information:
    # sum over events of weighted risk-set variance of z
    Iobs = mean( D.δ .* ( S2obs ./ S0obs .- (S1obs ./ S0obs).^2 ) )
    Uobs = step_eval( D.T, n.*cumsum(D.δ ./ S0obs.^2.0), t; left = 0.0)
    Vobs = step_eval( D.T, cumsum(D.δ .* S1obs ./ S0obs.^2.0), t; left = 0.0)
    return Iobs,  Vobs, Uobs
end
# Auxilliary function for `l` simulations of the asymptotic Cox NTR Bernstein-von Mises theorem Gaussian distributions in an array `t`
# ordered in an `l×length(t)` matrix
function Sfreq_mat(l::Int64,n::Int64,t::Vector{Float64},κbr::Vector{Float64},g::Vector{Float64},z::Vector{Float64},
            U::Function,V::Vector{Float64},I::Float64)
    d = length(g)
    #Sbr = exp.(-κbr)
    S_mats =  [ Matrix{eltype(t)}(undef, l, length(t)) for _ in 1:d ]
    for i in 1:l
        W = rand(Normal())
        Bu = BatU(t,U)
        for j in 1:d
            Sbr_j = exp.(-g[j] .* κbr)
            S_mats[j][i,:] = Sbr_j .* exp.( -g[j]*( ( V .- κbr.*z[j] ).*W./sqrt(I)  - Bu )  /sqrt(n) )
        end
    end
    return S_mats
end
# Auxilliary Linear predictors for Cox partial likelihood
cox_eta(c::Real, z::AbstractVector{<:Real}) = c * z[1]
cox_eta(c::AbstractVector{<:Real}, z::AbstractVector{<:Real}) = dot(c, z)
# Cox partial likelihood
function cox_partial_loglik_norep(c::Union{Real,AbstractVector{<:Real}}, D::RegressionSurvivalData)
    η = [ cox_eta(c, z) for z in D.Z]
    w = exp.(η)
    # Risk sums: R0[i] = sum_{j:T_j >= T_i} exp(c Z_j)
    R0 = reverse(cumsum(reverse(w)))
    return sum(D.δ .* (η .- log.(R0)))
end
function cox_partial_loglik_rep(c::Union{Real,AbstractVector{<:Real}}, D::RegressionSurvivalData)
    ll = 0.0
    R0 = 0.0
    @inbounds for j in D.n:-1:1
        for z in D.Zᵉ[j]
            R0 += exp( cox_eta(c, z) )
        end
        for z in D.Zᶜ[j]
            R0 += exp( cox_eta(c, z) )
        end
        d = D.nᵉ[j]
        if d > 0
            ll += sum( cox_eta(c, z) for z in D.Zᵉ[j] ) - d * log(R0)
        end
    end
    return ll
end
function cox_partial_loglik(c::Union{Real,AbstractVector{<:Real}}, D::RegressionSurvivalData)
    if hasproperty(D, :Zᵉ)
        return cox_partial_loglik_rep(c,D)
    else
        return cox_partial_loglik_norep(c,D)
    end
end
function bootstrap_survival(l::Int64, t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot_mat = zeros(l,length(t))
    for i in 1:l
        T_boot, H_boot, c_boot = boot_chaz( df, z_keys)
        S_boot = exp.( -exp( c_boot'*z_new ).*H_boot )
        S_boot_mat[i,:] = interp_surv(t,T_boot,S_boot)
    end   
    return S_boot_mat
end


function bootstrap_confidence_band( p::Float64, l::Int64,t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot =  bootstrap_survival( l, t, z_new, df, z_keys)
    return credible_band( p, S_boot)
end
