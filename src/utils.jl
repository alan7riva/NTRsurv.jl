function cred_band_mat(S::Array{Float64,2},n::Int64)
    for j in 1:n
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