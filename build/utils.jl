function surv_mat(l::Int64,t::Vector{Float64},model::ModelNTR)
    S_mat = zeros(Float64, length(t), l)
    for i in 1:l
        S_mat[:,i] = posterior_sim(t,model)
    end
    return S_mat
end

function cred_band_mat(S::Array{Float64,2},m::Int64)
    for _ in 1:m
        b = zeros(Float64, size(S)[1])
        for i in 1:size(S)[1]
            b[i] = maximum(S[i,:])-minimum(S[i,:])
        end
        b_max_index = sortperm(b)[end]
        s_out_index = findmax( abs.( S[b_max_index,:] .- mean(S[b_max_index,:]) ) )[2]
        S = S[1:end, 1:end .!= s_out_index ]
    end
    return S
end

function cred_band( p::Float64, l::Int64,t::Vector{Float64},model::ModelNTR,μ::Bool=true)
    if  !( 0 < p < 1)
        @error "ERROR: p is not between zero and one."
    end
    m = round( Int, l*p)
    S = surv_mat(l,t,model)
    for _ in 1:m
        b = zeros(Float64, size(S)[1])
        for i in 1:size(S)[1]
            b[i] = maximum(S[i,:])-minimum(S[i,:])
        end
        b_max_index = sortperm(b)[end]
        s_out_index = findmax( abs.( S[b_max_index,:] .- mean(S[b_max_index,:]) ) )[2]
        S = S[1:end, 1:end .!= s_out_index ]
    end
    band_u = [ maximum(S[i,:]) for i in 1:length(t)]
    band_d = [ minimum(S[i,:]) for i in 1:length(t)]
    band_m = zeros(length(t))
    if μ
        band_m = mean(S,dims=2)[1:end]
    else
        band_m = median(S,dims=2)[1:end];
    end
    return band_d, band_m, band_u
end