"""
    DataRegreNTRnorep

An immutable type containing possibly censored to the right observations with covariates and 
associated sufficient statistics, not depending on the Cox regression coefficients, for NTR Cox 
model fitting when there are no repetitions on the observations.
The type has the following fields:

- `T`: Sorted observation times.
- `δ`: Censoring indicators, 1 if exact observation and 0 otherwise, for sorted observation times `T`.
- `Z`: Covariates for sorted observation times `T`.
- `n`: Number of observations.
"""
struct DataRegreNTRnorep
    T::Vector{Float64} 
    δ::Vector{Int64}
    Z::Vector{Vector{Float64}} 
    n::Int64 
end

function DataRegreNTRnorep(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    sp = sortperm( T )
    T = T[ sp ]
    n = length(T)
    δ = δ[ sp ]
    Z = Z[ sp ]
    return DataRegreNTRnorep( T, δ, Z, n)
end

"""
    DataregreNTRrep

An immutable type containing possibly censored to the right observations with covariates and 
associated sufficient statistics, not depending on the Cox regression coefficients, for NTR Cox 
model fitting when there are no repetitions on the observations.
The type has the following fields:

- `To`: Sorted observation times.
- `T`: Sorted unique observation times.
- `δ`: Censoring indicators, 1 if exact observation and 0 otherwise, for sorted observation times `T`.
- `δᵉ`: Censoring indicators, 1 if exact observation is associated and 0 otherwise, for unique sorted observation times `T`.
- `δᶜ`: Censoring indicators, 1 if exact observation is associated and 0 otherwise, for unique sorted observation times `T`.
- `Z`: Covariates for sorted observation times `T`.
- `Zᵉ`: Covariates for sorted unique observation times `T` which are exactly observed, allowing for multiplicities.
- `Zᶜ`: Covariates for sorted unique observation times `T` which are not exactly observed, allowing for multiplicities.
- `n`: Number of observations.
- `m`: Number of unique observations.
- `nᵉ`: Frequencies of unique exact observations
"""

struct DataRegreNTRrep
    Tr::Vector{Float64}
    T::Vector{Float64}
    δr::Vector{Int64}
    δ::Vector{Int64}
    Z::Vector{Vector{Float64}}
    Zᵉ::Vector{Vector{Vector{Float64}}}
    Zᶜ::Vector{Vector{Vector{Float64}}}
    n::Int64
    m::Int64
    nᵉ::Vector{Int64}
end

function DataRegreNTRrep(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    n = length(T)
    sp = sortperm( T )
    T = T[ sp ]
    δ = δ[ sp ]
    Z = Z[ sp ]
    Tu = unique(T)
    m = length(Tu)
    Iᵉ = [ findall( (T .== v) .&& (δ .== 1.0) ) for v in unique(T) ]
    Iᶜ = [ findall( (T .== v) .&& (δ .== 0.0) ) for v in unique(T) ]
    Zᵉ = [ Z[v] for v in Iᵉ ]
    Zᶜ = [ Z[v] for v in Iᶜ ] 
    nᵉ = [ length(v) for v in Iᵉ ]
    nᶜ = [ length(v) for v in Iᶜ ]
    δᵉ = 1*( nᵉ .> 0 )
    return DataRegreNTRrep( T, Tu, δ, δᵉ, Z, Zᵉ, Zᶜ, n, m, nᵉ)
end

"""
    DataRegreNTR

Union type representing survival data objects for possibly censored to the right survival data with covariates in 
Cox NTR models.

`DataRegreNTR` is an alias for the union of internal data objects `DataRegreNTRnorep` and `DataRegreNTRrep`, corresponding respectively to datasets without and 
with repeated event times.
    
    DataRegreNTR(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})

Constructor for `DataNTR` with observed event times `T`, censoring indicators `δ` , where `δ[i] = 1` denotes an exact event and
`δ[i] = 0` denotes right censoring, and covariates Z.
"""
const DataRegreNTR = Union{DataRegreNTRnorep, DataRegreNTRrep}

function DataRegreNTR(T::Vector{Float64}, δ::Vector{Int64}, Z::Vector{Vector{Float64}})
    if minimum(T) < 0.0
        @error "Negative values in T are not supported for the data struct!"
    end
    if unique(T) != T
        return DataRegreNTRrep(T, δ, Z)
    else
        return DataRegreNTRnorep(T, δ, Z)
    end
end

"""
    BaselineRegreNTR

An immutable type for baseline setting of Cox regression neutral to the right (NTR) priors:

* `κ`: A priori cumulative hazard.
* `dκ`: A priori hazard rate. Needed for likelihood evaluations.
* `f`: Regression function
"""
"""
    BaselineRegreNTR

Immutable object for baseline specification of Cox NTR priors.

`BaselineRegreNTR` is specified by a Cox-baseline cumulative hazard function,
its hazard rate and optionally its inverse cumulative hazard.

    BaselineRegreNTR(b::BaselineNTR, f::Function)
    BaselineRegreNTR((b::BaselineNTR, f::Function)
    BaselineRegreNTR(v::Tuple{Function},f::Function)
    BaselineRegreNTR(v::Tuple{Function})

Missing `f` field is set to Cox regression specification by default. If either `v=(κ,dκ)` or `v=(κ,dκ,κinv)` are prvided 
then a BaselineNTR instantiated with those fields is used.

# Fields
- `b::BaselineNTR`: Cox-baseline.
- `f::Function`: Regressión function, if missing defaults to `f(c,z)=exp(c'*z)`.
"""
struct BaselineRegreNTR
    b::BaselineNTR
    f::Function
end

function BaselineRegreNTR(b::BaselineNTR)
    f(c::Vector{Float64},z::Vector{Float64}) = exp(c'*z) 
    return BaselineRegreNTR( b, f)
end

function BaselineRegreNTR(v::Tuple{Function},f::Function)
    b = BaselineNTR(v...)
    return BaselineRegreNTR( b, f)
end

function BaselineRegreNTR(v::Tuple{Function})
    b = BaselineNTR(v...)
    return BaselineRegreNTR( b)
end

"""
   SuffStatsRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for regression functions.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""

function SuffStatsRegreNTR(c::Vector{Float64},data::DataRegreNTRnorep,baseline::BaselineRegreNTR)
    n=data.n
    δ = data.δ
    Z = data.Z
    f = baseline.f
    hᵉ = [ (δ[i]==1) ? f(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of exact bservations
    hᶜ = [ (δ[i]==0) ? f(c,Z[i]) : 0.0 for i in 1:n ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    return R₁, R₂, hᵉ
end

function SuffStatsRegreNTR(c::Vector{Float64},data::DataRegreNTRrep,baseline::BaselineRegreNTR)
    m = data.m
    Zᵉ = data.Zᵉ
    Zᶜ = data.Zᶜ 
    f = baseline.f
    hᵉ = [ sum( [ f(c,v) for v in Zᵉ[i] ], init=0.0) for i in 1:m ] # frequencies of exact bservations
    hᶜ = [ sum( [ f(c,v) for v in Zᶜ[i] ], init=0.0) for i in 1:m ] # frequencies of censored observations
    Hᵉ = [ cumsum( hᵉ[end:-1:1] )[end:-1:1]; 0]
    Hᶜ = [ cumsum( hᶜ[end:-1:1] )[end:-1:1]; 0]
    R₁ = Hᵉ .+ Hᶜ 
    R₂ = Hᶜ .+ [ Hᵉ[2:end]; 0]
    F = [ [ [ length(v), sum( [ f(c,z) for z in Zᵉ[k][v]], init=0.0)] for v in collect(subsets(1:length(Zᵉ[k]))) ] for k in 1:m ]
    return R₁, R₂, F
end

"""
   loglikRegreNTR

Function for sufficient statistics in Cox regression NTR model. 

* `c`: Vector of parameters for Cox regression functions.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR baseline survival.
* `data`: Data struct for Cox regression NTR models, either type DataRegreNTRnorep or DataRegreNTRrep.
* `baseline`: Baseline struct for Cox regression NTR models.
"""

function loglikRegreNTR(c::Vector{Float64},α::Real,data::DataRegreNTRnorep,baseline::BaselineRegreNTR)
    l = 0.0
    κ = baseline.b.κ
    dκ = baseline.b.dκ
    β = 1.0/log(1.0+1.0/α)
    n = data.n
    X =  [0.0;data.T]
    R₁, R₂, hᵉ = SuffStatsRegreNTR(c,data,baseline)
    δ = data.δ
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( log( 1.0 + hᵉ[k]/(R₂[k]+α) ) )
    for k in 1:n
        l += cont_incr(k)
        if δ[k] == 1
            l += disc_incr(k)
        end
    end
    return l
end

function loglikRegreNTR(c::Vector{Float64},α::Real,data::DataRegreNTRrep,baseline::BaselineRegreNTR)
    l = 0.0
    κ = baseline.b.κ
    dκ = baseline.b.dκ
    f = baseline.f
    β = 1.0/log(1.0+1.0/α)
    m = data.m
    Zᵉ = data.Zᵉ
    X =  [0.0;data.T]
    R₁, R₂, F = SuffStatsRegreNTR(c,data,baseline)
    nᵉ = data.nᵉ
    cont_incr(k::Int64) = β*( κ(X[k+1])-κ(X[k]) )*log( α/(α + R₁[k]) )    
    disc_incr(k::Int64) = log( dκ(X[k+1]) ) + log(β) + log( sum( [ (-1.0)^(v[1]+1) * log( (R₂[k] + α + v[2])/α ) for v in F[k] ] ) )
    for k in 1:m
        l += cont_incr(k)
        if nᵉ[k] > 0
            l += disc_incr(k)
        end
    end
    return l
end

"""
    NTRmodelRegre

An immutable type for the NTR model framweork 
* `data`: Data struct with no repetitions in the obsevrations.
* `baseline`: Baseline struct for Cox regression NTR models.
* `c`: Vector of parameters for Cox regression functions.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
* `β`: Gamma process hyperparameter chosen for centering of NTR survival curves on baseline.
* `R₁`: Sufficient statistic for number of at risk observations after and including T_{(j)} factors.
* `R₂`: Sufficient statistic for number of at risk observations after T_{(j)} factors.
* `hᵉ`: Sufficient statistic for exact observation covariate factors.
"""

struct ModelRegreNTRnorep
    data::DataRegreNTRnorep
    baseline::BaselineRegreNTR
    c::Vector{Float64}
    α::Float64 
    β::Float64 
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    hᵉ::Vector{Float64}
end

struct ModelRegreNTRrep
    data::DataRegreNTRrep
    baseline::BaselineRegreNTR
    c::Vector{Float64}
    α::Float64
    β::Float64
    R₁::Vector{Float64}
    R₂::Vector{Float64}
    F::Vector{Vector{Vector{Float64}}}
end

const ModelRegreNTR = Union{ModelRegreNTRnorep, ModelRegreNTRrep}

function ModelRegreNTR(c::Vector{Float64},α::Float64,data::DataRegreNTRnorep,baseline::BaselineRegreNTR)
    β = 1.0/log(1.0+1.0/α)
    s1, s2, s3 = SuffStatsRegreNTR(c,data,baseline)
    return ModelRegreNTRnorep( data, baseline, c, α, β, s1, s2, s3)
end

function ModelRegreNTR(c::Vector{Float64},α::Float64,data::DataRegreNTRrep,baseline::BaselineRegreNTR)
    β = 1.0/log(1.0+1.0/α)
    s1, s2, s3 = SuffStatsRegreNTR(c,data,baseline)
    return ModelRegreNTRrep( data, baseline, c, α, β, s1, s2, s3)
end

"""
    postmeansurv

Function for posterior mean survival curve evaluation over a grid

* `t`: Time grid where posterior mean survival is evaluated.
* `data`: Data struct for NTR models, either type DataNTRnorep or DataNTRrep.
* `baseline`: Baseline struct for NTR models.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
* `β`: Gamma process hyperparameter chosen for centering of NTR survival curves on baseline.
"""

function postmeansurv(t::Vector{Float64},x_new::Vector{Float64},model::ModelRegreNTRnorep)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    κ = model.baseline.b.κ
    c = model.c
    α = model.α
    ν = model.baseline.f(model.c,x_new) 
    β = model.β
    n = model.data.n
    X =  [0.0;model.data.T]
    δ = model.data.δ
    hᵉ = model.hᵉ
    R₁ = model.R₁
    R₂ = model.R₂
    cont_incr(k::Int64) = exp( β*( κ(X[k])-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+ν) ) )
    cont_incr(k::Int64,t::Float64) = exp( β*( κ(t)-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+ν) ) )
    disc_incr(k::Int64) = log( (R₂[k]+α+ν+hᵉ[k])/(R₂[k]+α+ν) )/log( (R₂[k]+α+hᵉ[k])/(R₂[k]+α) )
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind =  t[i-1] .<= X[n_prev+1:end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> δ[j] == 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if δ[end] ==  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
    end
    return S
end

function postmeansurv(t::Vector{Float64},x_new::Vector{Float64},model::ModelRegreNTRrep)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    κ = model.baseline.b.κ
    c = model.c
    α = model.α
    ν = model.baseline.f(c,x_new) 
    β = model.β
    X =  [0.0;model.data.T]
    nᵉ = [model.data.nᵉ;0]
    R₁ = model.R₁
    R₂ = model.R₂
    F = model.F
    cont_incr(k::Int64) = exp( β*( κ(X[k])-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+ν) ) )
    cont_incr(k::Int64,t::Float64) = exp( β*( κ(t)-κ(X[k-1]) )*log( (α+R₁[k])/(α+R₁[k]+ν) ) )
    disc_incr(k::Int64) = sum( [ (-1.0)^(v[1]+1) * log( (R₂[k] + α + ν + v[2])/α ) for v in F[k] ] )/sum( [ (-1.0)^(v[1]+1) * log( (R₂[k] + α + v[2])/α ) for v in F[k] ] )
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind = t[i-1] .<= X[(n_prev+1):end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> nᵉ[j] >= 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if nᵉ[end] >=  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
    end
    return S
end

"""
   post_fix_locw_GammaNTR_accrej

Function for posterior simulation of weights at fixed locations corresponding to exact observations. 

* `l`: Number of simulaions from the vector of posterior weights.
* `data`: Data struct for NTR models, either type DataNTRnorep or DataNTRrep.
* `α`: Gamma process hyperparameter impacting Variance modulation for NTR survival curves.
"""

function post_fix_locw_GammaNTR_accrej(z_new::Vector{Float64},l::Int64,model::ModelRegreNTRnorep)
    n = model.data.n
    δ = model.data.δ
    e_bool = δ .== 1 
    m = sum(e_bool)
    W = zeros(m,l) 
    α = model.α
    ν = model.baseline.f(model.c,z_new) 
    R₂ = model.R₂
    hᵉ = model.hᵉ
    i_run = 1
    for i in collect(1:n)[ e_bool ]
        k = (α+R₂[i])/ν
        c = hᵉ[i]/ν
        for j in 1:l
            Y = rand(Gamma(1.0,1.0/k))
            logU = log(rand(Uniform()))
            while logU > log(1-exp(-c*Y)) - log(c*Y)
                Y = rand(Gamma(1.0,1.0/k))
                logU = log(rand(Uniform()))
            end
            W[i_run,j] = Y
        end
        i_run += 1
    end
    return W
end

function post_fix_locw_GammaNTR_accrej(z_new::Vector{Float64},l::Int64,model::ModelRegreNTRrep)
    n = model.data.m
    δ = model.data.δ
    e_bool = δ .== 1 
    m = sum(e_bool)
    W = zeros(m,l) 
    c = model.c
    α = model.α
    f = model.baseline.f
    ν = f(c,z_new)
    logν = log(ν)
    R₂ = model.R₂
    F = model.F
    i_run = 1
    for i in collect(1:n)[ e_bool ]
        k = (α+R₂[i])/ν
        nI = log(length( F[i] ))/log(2)
        logp = sum([ log(f(c,z)) for z in model.data.Zᵉ[i] ])
        for j in 1:l
            Y = rand(Gamma(nI,1.0/k))
            logU = log(rand(Uniform()))
            while logU > sum([ log(1.0 - exp( -f(c,z)*Y/ν)) for z in model.data.Zᵉ[i] ]) -logp  -nI*( log(Y) -logν )        
                Y = rand(Gamma(nI,1.0/k))
                logU = log(rand(Uniform()))
            end
            W[i_run,j] = Y
        end
        i_run += 1
    end
    return W
end

function post_fix_locw_GammaNTR_accrej(ν::Float64,i::Int64,model::ModelRegreNTRnorep)
    α = model.α
    R₂ = model.R₂
    hᵉ = model.hᵉ
    k = (α+R₂[i])/ν
    c = hᵉ[i]/ν
    Y = rand(Gamma(1.0,1.0/k))
    logU = log(rand(Uniform()))
    while logU > log(1-exp(-c*Y)) - log(c*Y)
        Y = rand(Gamma(1.0,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

function post_fix_locw_GammaNTR_accrej(ν::Float64,i::Int64,model::ModelRegreNTRrep)
    α = model.α
    f = model.baseline.f
    c = model.c
    logν = log(ν)
    R₂ = model.R₂
    F = model.F
    k = (α+R₂[i])/ν
    nI = log(length( F[i] ))/log(2)
    logp = sum([ log(f(c,z)) for z in model.data.Zᵉ[i] ])
    Y = rand(Gamma(nI,1.0/k))
    logU = log(rand(Uniform()))
    while logU > sum([ log(1.0 - exp( -f(c,z)*Y/ν)) for z in model.data.Zᵉ[i] ]) -logp  -nI*( log(Y) -logν )        
        Y = rand(Gamma(nI,1.0/k))
        logU = log(rand(Uniform()))
    end
    return Y
end

"""
   posterior_sim

Function for simulation of posterior survival curves in a grid of values using the analytical distribution of the increments.

* `t`: Time grid where posterior mean survival is evaluated.
* `model`: Model struct for NTR models.
"""

function posterior_sim(z_new::Array{Float64,1},t::Vector{Float64},model::ModelRegreNTR)
    if t[1] != 0.0
        t = [0.0;t]
    end
    S = [1.0]
    l = length(t)
    α = model.α
    β = model.β
    ν = model.baseline.f(model.c,z_new) 
    κ = model.baseline.b.κ
    X =  [0.0;model.data.T]
    δ = [model.data.δ;0]
    R₁ = model.R₁
    cont_incr(k::Int64) = exp( -rand(Gamma( β*(κ(X[k]) - κ(X[k-1])), 1/(α+R₁[k]+ν))) )
    cont_incr(k::Int64,t::Float64) = exp( -rand(Gamma( β*(κ(t) - κ(X[k-1])), 1/(α+R₁[k]+ν))) )
    disc_incr(k::Int64) = exp( -post_fix_locw_GammaNTR_accrej(ν,k,model) )
    cont_fact_run = 1.0
    n_prev = 1
    disc_fact_run = 1.0
    l_rec = findlast( t .< X[end] )
    for i in 2:l_rec
        X_inc_ind =  t[i-1] .<= X[n_prev+1:end] .< t[i] # indexes of observations which decrease survival between t[i-1] and t[i]
        n_inc = sum(X_inc_ind)
        if n_inc > 0
            n_forw = n_prev + n_inc
            cont_fact_run = cont_fact_run * mapreduce( j -> cont_incr(j),*,(n_prev+1):n_forw,init=1.0) # continuous part factor of decrease running by data observations, no mesh dependence
            disc_fact_run = disc_fact_run * mapreduce( j -> δ[j] == 1 ? disc_incr(j) : 1.0,*,(n_prev+1):n_forw,init=1.0) # discrete part factor of decrease running by data observations, no mesh dependence
            n_prev =  n_forw
        end
        push!( S, cont_fact_run*cont_incr(n_prev+1,t[i]) * disc_fact_run )
    end
    if l_rec < l
        cont_fact_run = cont_fact_run * cont_incr(n_prev+1,t[l_rec])
        if δ[end] >=  1
            disc_fact_run = disc_fact_run*disc_incr(n_prev+1)
        end
        for i in (l_rec+1):l
            push!( S, cont_fact_run*cont_incr(n_prev+1,t[i])*disc_fact_run )
        end
    end
    return S
end

"""
   posterior_sim

Function for simulation of posterior survival curves in a grid of values using the analytical distribution of the increments.

* `t`: Time grid where posterior mean survival is evaluated.
* `model`: Model struct for NTR models.
"""

function posterior_sim_prev(z_new::Array{Float64,1},model::ModelRegreNTR,t::Vector{Float64})
    X = model.data.T
    X0 = [0.0;X]
    Xend = X[end]
    n = length(t)
    t0 = copy(t)
    Y = zeros(n)
    if t[1] == 0.0
        t = t[2:end]
        n -= 1
    else
        t0 = [0.0;t]
    end
    i_first = findlast( t .<= X[1] )
    i_last = findlast( t .<= Xend )
    δ = model.data.δ
    c = model.c
    α = model.α
    ν = model.baseline.f(c,z_new) 
    β = model.β
    R₁ = model.R₁
    κ = model.baseline.b.κ
    W_fix = post_fix_locw_GammaNTR_accrej(z_new,1,model)
    n_fix = 0
    n_aux= 1
    if isnothing(i_first) 
        i_first = 0
    else
        X0[1] = t[i_first]
    end
    for i in 2:(i_first+1)
        cont_incr = rand(Gamma( β*(κ(t0[i]) - κ(t0[i-1])), 1/(α+R₁[i]+ν)))
        Y[i] = Y[i-1] + cont_incr
    end
    for i in (i_first+2):i_last
        ind  =  findall( X[n_aux:end] .<= t[i] ) .+ (n_aux -1)
        n_fix_incr =  sum( δ[ind], init=0)  + n_fix
        disc_incr = sum(W_fix[(n_fix+1):n_fix_incr], init=0.0)        
        cont_incr = sum( [ rand(Gamma( β*(κ(X[j]) - κ(X0[j])), 1/(α+R₁[j]+ν))) for j in ind ], init=0.0)
        Y[i] = Y[i-1] + disc_incr + cont_incr
        n_aux += length(ind)
        n_fix = n_fix_incr
    end  
    for i in (i_last+1):(n+1)
        cont_incr = rand(Gamma( β*(κ(t0[i]) - κ(t0[i-1])), 1/(α+ν)))
        Y[i] = Y[i-1] + cont_incr
    end
    return exp.(-Y)
end