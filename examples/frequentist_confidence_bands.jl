# In this file we include code for computation of fequentist confidence bands,
# which is used in the `uncertainty_quantification_with_cox_regression`` tutorial.
using Roots
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

# Function for equal precission confidence band computation

function ep_bands( km::KaplanMeier{Float64, Float64}, na::NelsonAalen{Float64, Float64}, qα::Float64)
    T = km_sim.events.time[end]
    qαT = sqrt(T)*qα
    n = sum(km.events.nevents)
    σ = sqrt.( n * u0_estim(km) )
    survival = km.survival
    chaz = na.chaz
    l = (qαT/sqrt(n)) .* σ ./ chaz
    lower = survival.^exp.( l )
    upper = survival.^exp.( -l )
    lower = clamp.(lower,0,1)
    upper = clamp.(upper,0,1)
    return lower[1:end-1], upper[1:end-1]
end

# Function for auxilliary computations of quantities needed for confidence bands in
# Cox regression models

function aux_comps( model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}} )
    β = coef(model)
    status = getproperty.(model.mf.data.event, :status)
    T = getproperty.(model.mf.data.event, :time)
    sp = sortperm(T)
    T = T[sp]
    status = status[sp]
    M = modelmatrix(model)
    n,p = size(M)
    η = M * β
    risk = exp.(η)
    # nS^(0)
    S0 = cumsum(risk[end:-1:1])[end:-1:1]
    # nS^(1)
    weightedZ = risk .* M
    S1 = similar(M)
    for j in 1:p
        S1[:,j] = cumsum(weightedZ[end:-1:1,j])[end:-1:1]
    end
    return β, risk, S0, S1, M, status, T
end

# Function for extimation of vector part in bilinear form of aymptotic variance 
# related to partial maximum likelihood estimation of regression coefficients
function cox_hvec_zbar_estim(t, z0, β, expβz0, risk, S0, S1, M, status, T)
    n,p = size(M)
    h = zeros(p)
    Zbar_v = zeros(n,p)
    for i in 1:n
        if status[i] == 1 && T[i] <= t
            Zbar_v[i,:] = S1[i,:] ./ S0[i]
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
function omega_estim( risk, S0, S1, M, status)
    n,p = size(M)
    Ω = zeros(p,p)
    S2 = zeros(p,p)
    for i in 1:n
        if status[i] == 1
            S0i = S0[i]
            S1i = S1[i,:]
            fill!(S2, 0.0)  # reuse allocation
            for j in i:n
                Mrow = @view M[j, :]
                S2 += risk[j] .* (Mrow * Mrow')
            end
            Ω += (S2 / S0i) - (S1i * S1i') / S0i^2
        end
    end
    # return nΩ and then account for Ω^{-1} = n (nΩ)^{-1}
    return Ω
end

# Function for estimation of asymptotic variance, accounting for varaince do to 
# Breslow estimator and maximum partial likelihood regression coefficients
function σ_estim( z0::Vector{Float64},model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β, risk, S0, S1, M, status, T = aux_comps(model)
    expβz0 = exp(dot(β,z0))
    n,p = size(M)
    l = length(T)
    Ω = omega_estim( risk, S0, S1, M, status)
    Ωinv = inv(Ω)
    σ² = zeros(l)
    for k in 1:l
        h, Zbar_v = cox_hvec_zbar_estim(T[k], z0, β, expβz0, risk, S0, S1, M, status, T)
        for i in 1:n
            if status[i] == 1 && T[i] <= T[k]
                σ²[k] += ( expβz0 / S0[i])^2 
            end
        end
        σ²[k] +=  h' * Ωinv * h
    end
    return sqrt.(n.*σ²)
end

function quantile_weighted_transformed_cox_wild_bootstrap(m::Int64,α::Float64,t::Vector{Float64},z0::Vector{Float64},model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}})
    β, risk, S0, S1, M, status, T = aux_comps(model)
    if minimum(t) < T[findfirst(status)]
        @error "ERROR: Evaluation array 't' starts befor first exact observation in data.."
    end
    expβz0 = exp(dot(β,z0))
    n,p = size(M)
    l = length(t)
    Z = model.mf.data.Z
    if p == 1
        Z = [ [z] for z in Z]
    end
    Ω = omega_estim( risk, S0, S1, M, status)
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
    println("Monte-Carlo quantile computation:")
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
    return cumsum( [ (δ[i]==1) ? 1.0./R[i] : 0.0 for i in 1:n] ), T 
end

function ep_bands( z0::Vector{Float64}, model::StatsModels.TableRegressionModel{CoxModel{Float64}, Matrix{Float64}}, qα::Float64)
    β = coef(model)
    H, T = BreslowEstimCoxModel(model)
    n = length(T)
    σ = σ_estim( z0, model)
    S = exp.( -H .* exp( β'*z0 ) )
    l = (qα/sqrt(n)) .* σ ./ H
    lower = S.^exp.( l )
    upper = S.^exp.( -l )
    lower = clamp.(lower,0,1)
    upper = clamp.(upper,0,1)
    return lower[1:end-1], upper[1:end-1], S, T
end