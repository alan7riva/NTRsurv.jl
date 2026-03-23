# In this file we include code for computation of fequentist confidence bands,
# which is used in the `uncertainty_quantification_with_cox_regression`` tutorial.
using Roots, ProgressMeter
import LinearAlgebra: dot

# Function for variance estimation of asymptotic band

function u0_estim( time::Vector{Float64}, n_risk::Vector{Int64}, n_event::Vector{Int64})
    v = zeros(length(time))
    acc = 0.0
    for i in eachindex(time)
        if n_event[i] > 0
            acc += n_event[i] / (n_risk[i]*(n_risk[i]-n_event[i]))
        end
        v[i] = acc
    end
    return v
end

function u0_estim(km::KaplanMeier{Float64, Float64})
    time = km.events.time
    n_risk = km.events.natrisk
    n_event = km.events.nevents
    return u0_estim( time, n_risk, n_event)
end

# Functions for quantile computation needed for equal precission band
# to be computed as
#p = 0.975
#qα = absmaxbrown_quantile(p,50)

function absmaxbrown_cdf(x,K=10)
    x <= 0 && return 0.0
    s = 0.0
    c = π^2 / (8x^2)
    sign = 1.0
    for k in 0:K
        m =  2k +1
        s += sign/m * exp(-m^2 * c)
        sign = -sign
    end
    return (4/π)*s
end

function absmaxbrown_quantile(p,K=10)
    f(x) = absmaxbrown_cdf(x,K) - p
    return find_zero(f, (0.1,5.0))
end

# With p = 0.95, absmaxbrown_quantile(p,50) returns 2.2414027273321393
#qα = 

ep_qα_precomps = [ 2.2414027273321393, 2.4977054744104645 ] 

# Function for equal precission confidence band computation

function ep_confidence_band( α::Float64, km::KaplanMeier{Float64, Float64}, na::NelsonAalen{Float64, Float64})
    qα = absmaxbrown_quantile(1-α,50)
    T = km.events.time[end]
    qαT = sqrt(T)*qα
    n = sum(km.events.nevents)
    σ² = n * u0_estim(km)
    σ = sqrt.( σ² )
    i_1 = findfirst( (σ² ./ (1 .+ σ²)) .>= 0.005 )
    i_2 = findlast( (σ² ./ (1 .+ σ²)) .<= 0.995 )
    survival = km.survival[i_1:i_2]
    chaz = na.chaz[i_1:i_2]
    l = (qαT/sqrt(n)) .* σ[i_1:i_2] ./ chaz
    lower = survival.^exp.( l )
    upper = survival.^exp.( -l )
    lower = clamp.(lower,0,1)
    upper = clamp.(upper,0,1)
    return lower, survival, upper, km.events.time[i_1:i_2]
end

# Function for auxilliary computations of quantities needed for confidence bands in
# Cox regression models
function aux_comps(model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β = coef(model)
    status = getproperty.(model.mf.data.event, :status)
    T = getproperty.(model.mf.data.event, :time)
    sp = sortperm(T)
    T = T[sp]
    status = status[sp]
    M = modelmatrix(model)
    n, p = size(M)
    η = M * β
    risk = exp.(η)
    # S^(0)
    S0 = cumsum(risk[end:-1:1])[end:-1:1]
    # S^(1)
    rM = eachrow(M)
    weightedZ = risk .* rM
    S1 = cumsum(weightedZ[end:-1:1])[end:-1:1]
    # S^(2)
    weightedZ2 = [ risk[i]*(rM[i] * rM[i]' ) for i in 1:n]
    S2 = cumsum(weightedZ2[end:-1:1])[end:-1:1]
    uT = unique(T)
    for t in uT
        idx = findall(T .== t)
        i0 = first(idx)
        risk[idx] .= risk[i0]
        S0[idx] .= S0[i0]
        S1[idx, :] .= S1[i0, :]
    end
    return β, risk, S0, S1, S2, M, status, T
end

# Function for extimation of vector part in bilinear form of aymptotic variance 
# related to partial maximum likelihood estimation of regression coefficients
function cox_hvec_zbar_estim(t, z0, β, expβz0, risk, S0, S1, M, status, T)
    n,p = size(M)
    h = zeros(p)
    Zbar_v = zeros(n,p)
    for i in 1:n
        if status[i] == 1 && T[i] <= t
            Zbar_v[i,:] = S1[i] / S0[i]
            w = (expβz0 / S0[i])
            for j in 1:p
                h[j] += w * (z0[j] - Zbar_v[i,j])
            end
        end
    end
    return h, Zbar_v
end

# Function for extimation of matrix part in bilinear form of aymptotic variance 
# related to partial maximum likelihood estimation of regression coefficients
function omega_estim( risk, S0, S1, S2, M, status)
    n,p = size(M)
    Ω = zeros(p,p)
    for i in 1:n
        if status[i] == 1
            S0i = S0[i]
            S1i = S1[i]
            S2i = S2[i]
            Ω += (S2i / S0i) - (S1i * S1i') / S0i^2
        end
    end
    # return nΩ and then account for Ω^{-1} = n (nΩ)^{-1}
    return Ω
end

# Function for estimation of asymptotic variance, accounting for varaince do to 
# Breslow estimator and maximum partial likelihood regression coefficients
function σ_estim( z0::Vector{Float64},model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β, risk, S0, S1, S2, M, status, T = aux_comps(model)
    uT = unique(T[status .== 1])
    expβz0 = exp(dot(β,z0))
    n,p = size(M)
    l = length(uT)
    Ω = omega_estim( risk, S0, S1, S2, M, status)
    Ωinv = inv(Ω)
    σ² = zeros(l)
    for k in 1:l
        h, _ = cox_hvec_zbar_estim(uT[k], z0, β, expβz0, risk, S0, S1, M, status, T)
        for i in 1:n
            if status[i] == 1 && T[i] <= T[k]
                σ²[k] += ( expβz0 / S0[i])^2 
            end
        end
        σ²[k] +=  h' * Ωinv * h
    end
    return sqrt.( σ²)
end

function quantile_weighted_transformed_cox_wild_bootstrap(m::Int64,α::Float64,t::Vector{Float64},z0::Vector{Float64},model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β, risk, S0, S1, S2, M, status, T = aux_comps(model)
    if minimum(t) < T[ findfirst(status .== 1 ) ]
        @error "ERROR: Evaluation array 't' starts befor first exact observation in data.."
    end
    expβz0 = exp(dot(β,z0))
    n,p = size(M)
    l = length(t)
    Z = [ collect(r) for r in eachrow(modelmatrix(model))]
    Ω = omega_estim( risk, S0, S1, S2, M, status)
    Ωinv = inv(Ω)
    v = zeros(m)
    u = zeros(l,n)
    w = zeros(l)
    σ² = zeros(l)
    for k in 1:l
        h, Zbar_v = cox_hvec_zbar_estim(t[k], z0, β, expβz0, risk, S0, S1, M, status, T)
        for i in 1:n
            f_tmp = h' * Ωinv
            if status[i] == 1
                u[k,i] += f_tmp*( Z[i] .- Zbar_v[i,:] )
                if T[i] <= t[k]
                    s0i_tmp = S0[i]
                    σ²[k] += ( expβz0 / s0i_tmp)^2 
                    u[k,i] +=  expβz0 / s0i_tmp
                end
            end
            σ²[k] +=  f_tmp*h
        end
    end
    println("Monte-Carlo quantile computation")
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for j in 1:m
        ϵ = rand( Normal(),l,n)
        v[j] = maximum( [ abs( sum( ϵ[i,:] .* u[i,:])/sqrt( σ²[i] ) ) for i in 1:l] )
        next!(prog)
    end
    return quantile(v,1-α)
end


function Breslow_estimator(model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β = coef(model)
    δ = Int.(getproperty.(model.mf.data.event, :status))
    T = getproperty.(model.mf.data.event, :time)
    sp = sortperm(T)
    T = T[sp]
    δ = δ[sp]
    M = modelmatrix(model)
    n,p = size(M)
    η = M * β
    r = exp.(η)
    R = cumsum( r[end:-1:1] )[end:-1:1]
    uT = unique(T)
    H = Float64[]
    times = Float64[]
    cumhaz = 0.0
    for t in uT
        idx = findall(T .== t)
        # number of events at time t
        d = sum(δ[idx])
        if d > 0
            # Breslow increment
            increment = d / R[first(idx)]
            cumhaz += increment
            push!(H, cumhaz)
            push!(times, t)
        end
    end
    return H, times
end

function ep_bands( z0::Vector{Float64}, model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}}, qα::Float64)
    β = coef(model)
    H, T = Breslow_estimator(model)
    n = length(T)
    σ =  sqrt(n) .* σ_estim( z0, model)
    S = exp.( -H .* exp( β'*z0 ) )
    l = (qα/sqrt(n)) .* σ ./ H
    lower = S.^exp.( l )
    upper = S.^exp.( -l )
    lower = clamp.(lower,0,1)
    upper = clamp.(upper,0,1)
    return lower[1:end-1], upper[1:end-1], S, T
end

function boot_chaz( df::DataFrame, z_keys::Vector{Symbol})
    n = nrow(df)
    inds = rand(1:n,n)
    freq_cox_model_boot = Survival.coxph( Term(:event) ~ sum(Term.(z_keys)) , df[inds, :]; tol=1e-8)
    c_boot = coef(freq_cox_model_boot)
    H_boot, T_boot = Breslow_estimator(freq_cox_model_boot)
    return T_boot, H_boot, c_boot
end

function interp_surv( t::Float64, T::Vector{Float64}, S::Vector{Float64})
    i = searchsortedlast(T, t)
    return i == 0 ? 1.0 : S[i]
end

function interp_surv( t::Vector{Float64}, T::Vector{Float64}, S::Vector{Float64})
    return [interp_surv( ti, T, S) for ti in t]
end

function bootstrap_survival(l::Int64,t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot_mat = zeros(l,length(t))
    for i in 1:l
        T_boot, H_boot, c_boot = boot_chaz( df, z_keys)
        S_boot = exp.( -exp( c_boot'*z_new ).*H_boot )
        S_boot_mat[i,:] = interp_surv(t,T_boot,S_boot)
    end   
    return S_boot_mat
end

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

function bootstrap_credible_band( p::Float64, l::Int64,t::Vector{Float64}, z_new::Vector{Float64}, df::DataFrame, z_keys::Vector{Symbol})
    S_boot =  bootstrap_survival( l, t, z_new, df, z_keys)
    return credible_band( p, S_boot)
end