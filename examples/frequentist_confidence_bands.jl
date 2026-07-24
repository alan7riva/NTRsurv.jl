# In this file we include code for computation of fequentist confidence bands,
# which is used in some of the examples folder tutorials.
import LinearAlgebra: dot, cholesky, Symmetric
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

# Breslow estimator for cumulative hazard in frequentist Cox models, with times as output
function BreslowEstimCoxModel_with_time(model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    event = model.mf.data.event
    T = Float64.(getfield.(event, :time))
    δ = Int64.(getfield.(event, :status))
    c = coef(model)
    M = modelmatrix(model)
    η = M * c
    risk = exp.(η)
    sp = sortperm(T)
    T = T[sp]
    δ = δ[sp]
    risk = risk[sp]
    Tu = unique(T)
    d = zeros(Int64, length(Tu))
    R = zeros(Float64, length(Tu))
    for j in eachindex(Tu)
        tj = Tu[j]
        # Events at tj
        d[j] = sum(δ[T .== tj])
        # Risk set just before tj: all observations with T_i >= tj
        R[j] = sum(risk[T .>= tj])
    end
    H = cumsum(d ./ R)
    return H, Tu
end

# Breslow estimator for cumulative hazard in frequentist Cox models
function BreslowEstimCoxModel(model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    H, _ = BreslowEstimCoxModel_with_time(model)
    return H
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
# Auxilliary Linear predictors for Cox partial likelihood
cox_eta(c::Real, z::AbstractVector{<:Real}) = c * z[1]
cox_eta(c::AbstractVector{<:Real}, z::AbstractVector{<:Real}) = dot(c, z)
cox_dim(c::Real) = 1
cox_dim(c::AbstractVector{<:Real}) = length(c)


function step_eval_columns(x::Vector{Float64}, Y::Matrix{Float64}, q::Vector{Float64}; left::Float64 = 0.0)
    return hcat([
        step_eval(x, Y[:,j], q; left = left)
        for j in 1:size(Y,2)
    ]...)
end

function cox_var_obs_norep(t::Vector{Float64},c::Float64,D::RegressionSurvivalData)
    n = length(D.Z)
    η = [cox_eta(c, z) for z in D.Z]
    w = exp.(η)
    Zv = [z[1] for z in D.Z]
    # Risk sums: R_i(c) = sum_{j:T_j >= T_i} exp(c Z_j)
    S0obs = reverse(cumsum(reverse(w)))
    S1obs = reverse(cumsum(reverse(w .* Zv)))
    S2obs = reverse(cumsum(reverse(w .* Zv.^2.0)))
    # Observed Fisher information per observation:
    # I_n(c) = (1/n) sum_i δ_i [S2/S0 - (S1/S0)^2]
    Iobs = sum(D.δ .* (S2obs ./ S0obs .- (S1obs ./ S0obs).^2)) / n
    # Observed Brownian time-change:
     # U_n(t) = n sum_{T_i <= t} δ_i / S0obs[i]^2
    Uobs = step_eval( D.T, n .* cumsum(D.δ ./ S0obs.^2.0), t; left = 0.0)
    # Observed cumulative regression-coupling term:
    # V_n(t) = sum_{T_i <= t} δ_i S1obs[i] / S0obs[i]^2
    Vobs = step_eval( D.T, cumsum(D.δ .* S1obs ./ S0obs.^2.0), t; left = 0.0)
    return Iobs, Vobs, Uobs
end

function cox_var_obs_norep(t::Vector{Float64},c::Vector{Float64},D::RegressionSurvivalData)
    n = length(D.Z)
    p = cox_dim(c)
    η = [cox_eta(c, z) for z in D.Z]
    w = exp.(η)
    Zv = zeros(Float64, n, p)
    for i in 1:n
        Zv[i, :] .= D.Z[i]
    end
    # Risk sums:
    # S0obs[i]    = sum_{j:T_j >= T_i} exp(c'Z_j)
    # S1obs[i, :] = sum_{j:T_j >= T_i} Z_j exp(c'Z_j)
    S0obs = reverse(cumsum(reverse(w)))
    S1_contrib = Zv .* reshape(w, :, 1)
    S1obs = reverse( cumsum( reverse(S1_contrib, dims = 1), dims = 1), dims = 1)
    # Matrix-valued risk sum:
    # S2obs[i, :, :] = sum_{j:T_j >= T_i} Z_j Z_j' exp(c'Z_j)
    S2obs = zeros(Float64, n, p, p)
    for a in 1:p
        for b in 1:p
            S2obs[:,a,b] .= reverse( cumsum( reverse(w .* Zv[:,a] .* Zv[:,b]) ) )
        end
    end
    # Observed Fisher information matrix per observation:
    # I_n(c) = (1/n) sum_i δ_i [S2/S0 - (S1/S0)(S1/S0)']
    Iobs = zeros(Float64, p, p)
    for i in 1:n
        if D.δ[i] == 1
            s0 = S0obs[i]
            s1 = vec(S1obs[i, :])
            s2 = S2obs[i, :, :]
            Iobs .+= (s2 ./ s0 .- (s1 * s1') ./ s0^2) ./ n
        end
    end
    # Observed Brownian time-change:
    # U_n(t) = n sum_{T_i <= t} δ_i / S0obs[i]^2
    Uobs = step_eval( D.T, n .* cumsum(D.δ ./ S0obs.^2.0), t; left = 0.0 )
    # Observed cumulative regression-coupling term:
    # V_n(t) = sum_{T_i <= t} δ_i S1obs[i, :] / S0obs[i]^2
    V_incr = S1obs .* reshape(D.δ ./ S0obs.^2.0, :, 1)
    V_event = cumsum(V_incr, dims = 1)
    Vobs = step_eval_columns( D.T, V_event, t; left = 0.0)
    return Iobs, Vobs, Uobs
end

function cox_var_obs_rep(t::Vector{Float64}, c::Float64, D::RegressionSurvivalData)
    n = length(D.Z)
    m = D.n # number of unique observed times
    Wblock = zeros(Float64, m)
    ZWblock = zeros(Float64, m)
    Z2Wblock = zeros(Float64, m)
    # Contributions at each unique observed time
    for j in 1:m
        for z in D.Zᵉ[j]
            zi = z[1]
            wi = exp(cox_eta(c, z))
            Wblock[j] += wi
            ZWblock[j] += wi * zi
            Z2Wblock[j] += wi * zi^2
        end
        for z in D.Zᶜ[j]
            zi = z[1]
            wi = exp(cox_eta(c, z))
            Wblock[j] += wi
            ZWblock[j] += wi * zi
            Z2Wblock[j] += wi * zi^2
        end
    end
    # Risk sums:
    # S0obs[j] = sum_{i:T_i >= τ_j} exp(c Z_i)
    # S1obs[j] = sum_{i:T_i >= τ_j} Z_i exp(c Z_i)
    # S2obs[j] = sum_{i:T_i >= τ_j} Z_i^2 exp(c Z_i)
    S0obs = reverse(cumsum(reverse(Wblock)))
    S1obs = reverse(cumsum(reverse(ZWblock)))
    S2obs = reverse(cumsum(reverse(Z2Wblock)))
    # Observed Fisher information per observation:
    # I_n(c) = (1/n) sum_j d_j [S2/S0 - (S1/S0)^2]
    Iobs = sum(D.nᵉ .* (S2obs ./ S0obs .- (S1obs ./ S0obs).^2)) / n
    # Observed Brownian time-change:
    # U_n(t) = n sum_{τ_j <= t} d_j / S0obs[j]^2
    Uobs = step_eval( D.T, n .* cumsum(D.nᵉ ./ S0obs.^2.0), t; left = 0.0)
    # Observed cumulative regression-coupling term:
    # V_n(t) = sum_{τ_j <= t} d_j S1obs[j] / S0obs[j]^2
    Vobs = step_eval( D.T, cumsum(D.nᵉ .* S1obs ./ S0obs.^2.0), t; left = 0.0)
    return Iobs, Vobs, Uobs
end

function cox_var_obs_rep(t::Vector{Float64}, c::AbstractVector{<:Real}, D::RegressionSurvivalData)
    n = length(D.Z)
    m = D.n # number of unique observed times
    p = length(c)
    Wblock = zeros(Float64, m)
    ZWblock = zeros(Float64, m, p)
    Z2Wblock = zeros(Float64, m, p, p)
    # Contributions at each unique observed time
    for j in 1:m
        for z in D.Zᵉ[j]
            wi = exp(cox_eta(c, z))
            Wblock[j] += wi
            ZWblock[j, :] .+= wi .* z
            for a in 1:p
                for b in 1:p
                    Z2Wblock[j,a,b] += wi * z[a] * z[b]
                end
            end
        end
        for z in D.Zᶜ[j]
            wi = exp(cox_eta(c, z))
            Wblock[j] += wi
            ZWblock[j, :] .+= wi .* z
            for a in 1:p
                for b in 1:p
                    Z2Wblock[j,a,b] += wi * z[a] * z[b]
                end
            end
        end
    end
    # Risk sums:
    # S0obs[j]      = sum_{i:T_i >= τ_j} exp(c'Z_i)
    # S1obs[j, :]   = sum_{i:T_i >= τ_j} Z_i exp(c'Z_i)
    # S2obs[j, :, :] = sum_{i:T_i >= τ_j} Z_i Z_i' exp(c'Z_i)
    S0obs = reverse(cumsum(reverse(Wblock)))
    S1obs = reverse( cumsum( reverse(ZWblock, dims = 1), dims = 1), dims = 1)
    S2obs = reverse( cumsum( reverse(Z2Wblock, dims = 1), dims = 1), dims = 1)
    # Observed Fisher information matrix per observation:
    # I_n(c) = (1/n) sum_j d_j [S2/S0 - (S1/S0)(S1/S0)']
    Iobs = zeros(Float64, p, p)
    for j in 1:m
        if D.nᵉ[j] > 0
            s0 = S0obs[j]
            s1 = vec(S1obs[j, :])
            s2 = S2obs[j, :, :]
            Iobs .+= (D.nᵉ[j] / n) .* (s2 ./ s0 .- (s1 * s1') ./ s0^2)
        end
    end
    # Observed Brownian time-change:
    # U_n(t) = n sum_{τ_j <= t} d_j / S0obs[j]^2
    Uobs = step_eval( D.T, n .* cumsum(D.nᵉ ./ S0obs.^2.0), t; left = 0.0)
    # Observed cumulative regression-coupling term:
    # V_n(t) = sum_{τ_j <= t} d_j S1obs[j, :] / S0obs[j]^2
    V_incr = S1obs .* reshape(D.nᵉ ./ S0obs.^2.0, :, 1)
    V_event = cumsum(V_incr, dims = 1)
    Vobs = step_eval_columns( D.T, V_event, t; left = 0.0)
    return Iobs, Vobs, Uobs
end
function cox_var_obs(t::Vector{Float64}, c::Union{Real,AbstractVector{<:Real}}, D::RegressionSurvivalData)
    if hasproperty(D, :Zᵉ)
        return cox_var_obs_rep(t, c, D)
    else
        return cox_var_obs_norep(t, c, D)
    end
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

function Sfreq_mat( l::Int64, n::Int64, t::Vector{Float64}, κbr::Vector{Float64}, g::Vector{Float64},
    z::Vector{Vector{Float64}}, U::Function, V::Matrix{Float64}, I::Matrix{Float64})
    d = length(g)
    p = size(I, 1)
    #@assert size(I, 1) == size(I, 2)
    #@assert size(V, 1) == length(t)
    #@assert size(V, 2) == p
    #@assert all(length.(z) .== p)
    # Cholesky factor for I^{-1/2}W.
    # If I = U'U, then U \ W has covariance I^{-1}.
    I_chol = cholesky(Symmetric(Matrix(I))).U
    S_mats = [ Matrix{eltype(t)}(undef, l, length(t)) for _ in 1:d ]
    for i in 1:l
        W = randn(p)
        X = I_chol \ W
        Bu = BatU(t, U)
        VX = V * X
        for j in 1:d
            Sbr_j = exp.(-g[j] .* κbr)
            # Rowwise value of (V(t) - κbr(t) z_j)' X
            reg_j = VX .- κbr .* dot(z[j], X)
            S_mats[j][i,:] = Sbr_j .* exp.( g[j] .* (reg_j .- Bu) ./ sqrt(n) )
        end
    end
    return S_mats
end

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

function boot_chaz( df::DataFrame, z_keys::Vector{Symbol})
    n = nrow(df)
    inds = rand(1:n,n)
    df_boot = df[inds,:]
    freq_cox_model_boot = Survival.coxph( Term(:event) ~ sum(Term.(z_keys)) , df_boot; tol=1e-8)
    c_boot = coef(freq_cox_model_boot)
    H_boot, T_boot = BreslowEstimCoxModel_with_time( freq_cox_model_boot)
    return T_boot, H_boot, c_boot
end

function bootstrap_survival(l::Int64, t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot_mat = zeros(l,length(t))
    for i in 1:l
        T_boot, H_boot, c_boot = boot_chaz( df, z_keys)
        H_boot_t = step_eval( T_boot, H_boot, t;left = 0.0)
        S_boot_mat[i, :] = exp.( -exp(dot(c_boot, z_new)) .* H_boot_t)
    end   
    return S_boot_mat
end


function bootstrap_confidence_band( p::Float64, l::Int64,t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot =  bootstrap_survival( l, t, z_new, df, z_keys)
    return credible_band( p, S_boot)
end
