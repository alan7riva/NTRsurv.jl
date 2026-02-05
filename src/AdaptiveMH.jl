@inline function RandWalkMHstep( lf::Function, s::Float64, lfs::Float64, prop_σ::Float64)
    sprop = s + rand( Normal(0.0, prop_σ) )
    lfsprop = lf( sprop )
    u = rand(Uniform())
    if log(u) < min( lfsprop - lfs, 0.0)
        s = sprop
        lfs = lfsprop
    end
    return s, lfs
end

@inline function RandWalkMHwithinGIBBSstep( lf::Function, d::Int64, s::Vector{Float64}, lfs::Float64, prop_σ::Vector{Float64})
    for i in 1:d
        s[i], lfs = RandWalkMHstep( x -> lf( [s[1:(i-1)];x;s[(i+1):d]] ), s[i], lfs, prop_σ[i])
    end
    return s, lfs
end

@inline function RandWalkMHwithinBlockedGIBBSstep( lf::Vector{Function}, blocks::Vector{UnitRange{Int64}}, l::Int64, d::Vector{Int64}, s::Vector{Float64}, lfs::Vector{Float64}, prop_σ::Vector{Float64})
    for i in 1:l
        s_block = s[ blocks[i] ]
        prop_σ_block = prop_σ[ blocks[i] ]
        for j in 1:d[i]
            s[ blocks[i][j] ], lfs[i] = RandWalkMHstep( x -> lf[i]( [s_block[1:(j-1)];x;s_block[(j+1):d[i]]] ), s_block[j], lfs[i], prop_σ_block[j])
        end
    end
    return s, lfs
end

@inline function RandWalkMH( b::Int64, lf::Function, s::Float64, lfs::Float64, prop_σ::Float64)
    chain_s = zeros(b)
    chain_s[1] = s
    s_run = copy(s)
    lfs_run = lfs
    for j in 2:b
        s_run, lfs_run = RandWalkMHstep( lf, s_run, lfs_run, prop_σ)
        chain_s[j] = copy(s_run)
    end
    return chain_s, lfs_run
end

@inline function RandWalkMHwithinGibbs( b::Int64, lf::Function, s::Vector{Float64}, lfs::Float64, prop_σ::Vector{Float64})
    d = length(s)
    chain_s = [ zeros(d) for _ in 1:b ]
    chain_s[1] = s
    s_run = copy(s)
    lfs_run = lfs
    for j in 2:b
        s_run, lfs_run = RandWalkMHwithinGIBBSstep( lf, d, s_run, lfs_run, prop_σ)
        chain_s[j] = copy(s_run)
    end
    return chain_s, lfs_run
end

@inline function RandWalkMHwithinBlockedGibbs( b::Int64, lf::Vector{Function}, blocks::Vector{UnitRange{Int64}}, s::Vector{Float64}, lfs::Vector{Float64}, prop_σ::Vector{Float64})
    l = length(blocks)
    d = length.(blocks)
    chain_s = [ zeros(length(s)) for _ in 1:b ]
    chain_s[1] = s
    s_run = copy(s)
    lfs_run = lfs
    for j in 2:b
        s_run, lfs_run = RandWalkMHwithinBlockedGIBBSstep( lf, blocks, l, d, s_run, lfs_run, prop_σ)
        chain_s[j] = copy(s_run)
    end
    return chain_s, lfs_run
end

function MCMCchainAcc(v::Vector{Float64})
    l = length(v)
    rej = length( findall( j-> v[j+1] != v[j], collect(1:(l-1)) ) )
    return rej/(l-1)
end

function MCMCchainAcc(v::Vector{Vector{Float64}})
    l = length(v)
    d = length(v[1])
    rej = zeros(d)
    for i in 1:d
        rej[i] = length( findall( j-> v[j+1][i] != v[j][i], collect(1:(l-1)) ) )
    end
    return rej/(l-1)
end

# n <- iteraciones de Robbins-Monro
# b <- Robbins-Monro batch size, in this case Metropolis-Hastings steps
# lf <- loglikelihood with constant terms substracted
# s₀ <- overall initial state for the Metropolis-Hastings algoritm
# σ₀ <- initial state for Robbins-Monro algorithm pertaining to the standard deviations in Metropolis- Hastings algorithm
# p_acc <- targeted probability of acceptance in Robbins-Monro algorithm
# γ <- learning rate of Robbins-Monro algorithm 
@inline function RobMonMHtuneWithProg(n::Int64,b::Int64,lf::Function,s₀::Float64,σ₀::Float64,
        p_acc::Float64,γ::Float64)
    s_run = s₀
    lfs_run = lf(s₀)
    c_σ = zeros(n+1) 
    c_θ = zeros(n+1) 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    # Progress meter
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMH( b, lf, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] + (1/i^γ)*( MCMCchainAcc(c_s_tmp) - p_acc )
        c_σ[i] = exp(c_θ[i])
        next!(prog)
    end
    return c_σ, s_run, lfs_run
end

@inline function RobMonMHtuneWithoutProg(n::Int64,b::Int64,lf::Function,s₀::Float64,σ₀::Float64,
        p_acc::Float64,γ::Float64)
    s_run = s₀
    lfs_run = lf(s₀)
    c_σ = zeros(n+1) 
    c_θ = zeros(n+1) 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMH( b, lf, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] .+ (1/i^γ).*( MCMCchainAcc(c_s_tmp) .- p_acc )
        c_σ[i] = exp.(c_θ[i])
    end
    return c_σ, s_run, lfs_run
end

function RobMonMHtune(n::Int64,b::Int64,lf::Function,s₀::Float64,σ₀::Float64,
        p_acc::Float64,γ::Float64,show_progress::Bool=true)
    if show_progress
        return RobMonMHtuneWithProg(n,b,lf,s₀,σ₀,p_acc,γ)
    else
        return RobMonMHtuneWithoutProg(n,b,lf,s₀,σ₀,p_acc,γ)
    end
end

@inline function RobMonMHwithinGIBBStuneWithProg(n::Int64,b::Int64,lf::Function,s₀::Vector{Float64},σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64)
    d = length(s₀)
    s_run = s₀
    lfs_run = lf(s₀)
    c_σ = [ zeros(d) for _ in 1:(n+1) ] 
    c_θ = [ zeros(d) for _ in 1:(n+1) ] 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    # Progress meter
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMHwithinGibbs( b, lf, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] .+ (1/i^γ).*( MCMCchainAcc(c_s_tmp) .- p_acc )
        c_σ[i] = exp.(c_θ[i])
        next!(prog)
    end
    return c_σ, s_run, lfs_run
end

@inline function RobMonMHwithinGIBBStuneWithoutProg(n::Int64,b::Int64,lf::Function,s₀::Vector{Float64},σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64)
    d = length(s₀)
    s_run = s₀
    lfs_run = lf(s₀)
    c_σ = [ zeros(d) for _ in 1:(n+1) ] 
    c_θ = [ zeros(d) for _ in 1:(n+1) ] 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMHwithinGibbs( b, lf, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] .+ (1/i^γ).*( MCMCchainAcc(c_s_tmp) .- p_acc )
        c_σ[i] = exp.(c_θ[i])
    end
    return c_σ, s_run, lfs_run
end

function RobMonMHwithinGIBBStune(n::Int64,b::Int64,lf::Function,s₀::Vector{Float64},σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64,show_progress::Bool=true)
    if show_progress
        return RobMonMHwithinGIBBStuneWithProg(n,b,lf,s₀,σ₀,p_acc,γ)
    else
        return RobMonMHwithinGIBBStuneWithoutProg(n,b,lf,s₀,σ₀,p_acc,γ)
    end
end

@inline function RobMonMHwithinBlockedGIBBStuneWithProg( n::Int64, b::Int64, lf::Vector{Function}, blocks::Vector{UnitRange{Int64}}, s₀::Vector{Float64}, σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64)
    d = length(s₀)
    s_run = s₀
    s₀_blocks = [ s₀[i] for i in blocks ]
    lfs_run = [ lf[i]( s₀_blocks[i] ) for i in 1:length(blocks) ]
    c_σ = [ zeros(d) for _ in 1:(n+1) ] 
    c_θ = [ zeros(d) for _ in 1:(n+1) ] 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    # Progress meter
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMHwithinBlockedGibbs( b, lf, blocks, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] .+ (1/i^γ).*( MCMCchainAcc(c_s_tmp) .- p_acc )
        c_σ[i] = exp.(c_θ[i])
        next!(prog)
    end
    return c_σ, s_run, lfs_run
end

@inline function RobMonMHwithinGIBBStuneWithoutProg( n::Int64, b::Int64, lf::Vector{Function}, blocks::Vector{UnitRange{Int64}}, s₀::Vector{Float64}, σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64)
    d = length(s₀)
    s_run = s₀
    s₀_blocks = [ s₀[i] for i in blocks ]
    lfs_run = [ lf[i]( s₀_blocks[i] ) for i in 1:length(blocks) ]
    c_σ = [ zeros(d) for _ in 1:(n+1) ] 
    c_θ = [ zeros(d) for _ in 1:(n+1) ] 
    c_σ[1] =  σ₀ 
    c_θ[1] = log.(σ₀)
    for i in 2:(n+1)
        c_s_tmp, lfs_run = RandWalkMHwithinGibbs( b, lf, s_run, lfs_run, c_σ[i-1])
        s_run = c_s_tmp[end]
        c_θ[i] = c_θ[i-1] .+ (1/i^γ).*( MCMCchainAcc(c_s_tmp) .- p_acc )
        c_σ[i] = exp.(c_θ[i])
    end
    return c_σ, s_run, lfs_run
end

function RobMonMHwithinBlockedGIBBStune( n::Int64, b::Int64, lf::Vector{Function}, blocks::Vector{UnitRange{Int64}}, s₀::Vector{Float64},σ₀::Vector{Float64},
        p_acc::Vector{Float64},γ::Float64,show_progress::Bool=true)
    if show_progress
        return RobMonMHwithinBlockedGIBBStuneWithProg( n, b, lf, blocks, s₀, σ₀, p_acc, γ)
    else
        return RobMonMHwithinBlockedGIBBStuneWithoutProg( n, b, lf, blocks, s₀, σ₀, p_acc, γ)
    end
end
