


function sgd{T<:FloatingPoint}(nn::NN, λ0, μ, 
    X::Array{T}, 
    Y::Array{T};
    subset=1:size(X,2), 
    minibatch=100, 
    steps=100, 
    freq=0,
    weigths=0, 
    metric::PreMetric=SqEuclidean())

    nsamples = size(X,2)
    @assert size(Y,2) == nsamples "input and output do not have the same number of samples"
    @assert minibatch < nsamples "minibatch size should be smaller than number of samples"

    state = TrainState(nn, minibatch)
    mdws = Array[ zeros(w) for w in nn.ws ] # momentum on weight gradient
    mdbs = Array[ zeros(w) for w in nn.ws ] # momentum on bias   gradient

    # (subset==0) && (subset = shuffle(collect(1:size(X,2))))

    resid = nsamples % minibatch
    resid > 0 && 
        warn("# samples not a multiple of minibatch size, $resid tail samples will be ignored")

    pos = 1 # pos = 47501
    for k in 0:steps-1
        if pos + minibatch > nsamples
            rg = 1:minibatch
            pos = minibatch + 1
        else 
            rg = pos:pos+minibatch-1
            pos += minibatch
        end
        # if pos + minibatch > length(subset)
        #     rg = subset[pos:end]
        #     pos = pos + minibatch - length(subset)
        #     rg = vcat(rg, subset[1:pos-1])
        # else 
        #     rg = subset[pos:pos+minibatch-1]
        #     pos += minibatch
        # end

        # cycle!(state, nn, X[:,rg], Y[:,rg], metric=metric)
        cycle!(state, nn, sub(X,:,rg), sub(Y,:,rg), metric=metric)
        # isnan(state, out[1]) && break

        l2norm = sqrt(sum( map(sumabs2, state.dws ) ) +
                      sum( map(sumabs2, state.dbs ) ) )
        λ    = - λ0 * (l2norm > 1. ? 1. / l2norm : 1.)

        for j in 1:depth(nn) # j = 1
            for l in 1:length(nn.ws[j])
                tmp = λ * (1-μ) * state.dws[j][l] + μ * mdws[j][l]
                nn.ws[j][l] += tmp
                mdws[j][l] = tmp
            end
            for l in 1:length(nn.bs[j])
                tmp = λ * (1-μ) * state.dbs[j][l] + μ * mdbs[j][l]
                nn.bs[j][l] += tmp
                mdbs[j][l] = tmp
            end
        end

        if freq > 0 && (k % freq == 0)
            pred = calc(nn, X)
            m = mean(pred - Y)
            s = std(pred - Y)
            println("k : $k λ : $(round(λ0,4)) mean : $(round(m,5)) stderr : $(round(s,5))")
        end
    end

    # score(f, ls, subset, vars, res, pond)
end

# function pathsgd(f, df, ls, subset, λ0, μ, vars, res, pond; 
#     steps=100, freq=0, noise=0., dropout=0.)

#     ns   = Int[ Int[ size(w,2) for w in ls ] ; size(ls[end],1) ]

#     # allocations
#     dls   = Array{Float64}[ zeros(w) for w in ls]
#     mask  = Array{Float64}[ ones(w) for w in ls]
#     hmask = Array{Bool}[ ones(l) for l in ns]
#     mls   = Array{Float64}[ Array(Float64, size(w)) for w in ls]

#     nw   = Array{Float64}[ Array(Float64, size(w)) for w in ls]
#     γin  = Array{Float64}[ Array(Float64, l) for l in ns ]
#     γout = Array{Float64}[ Array(Float64, 1, l) for l in ns ]

#     ri = 1
#     for k in 0:steps-1
#         i = subset[ri]  # i = idx[ri]

#         ### dropout + calc of gradient
#             x  = vec( vars[i,:] )
#             noise != 0. && ( x += randn(length(x))*noise )
#             # dropout = 0.3
#             if dropout != 0.
#                 for l in 1:length(ns)-1 # last (output node) is always true 
#                     for j in 1:length(hmask[l])
#                         hmask[l][j] = rand() .> dropout
#                     end
#                 end

#                 for l in 1:length(mask) # l = 1
#                     A_mul_B!(mask[l], hmask[l+1], hmask[l]')
#                     for j in 1:length(mask[l])
#                         mls[l][j] = ls[l][j] * mask[l][j]
#                     end
#                 end

#                 out = df(mls..., x) # out = dann(mls..., x)
#             else
#                 out = df(ls..., x) # out = dann(ls..., x)
#             end
            
#             if isnan(out[1])
#                 println(mls)
#                 println(dls)
#                 return
#             end

#         ### gradient step normalization for path-SGD
#             p = 2.
#             depth = length(ls)
#             copy!(γin[1], hmask[1])  #fill!(γin[1], 1.0)
#             fill!(γout[end], 1.0)
#             # calc abs(weight) ^ p, store temporarily in nw
#             for (l,w) in enumerate(ls)  # l = 1 ; w=ls[1]
#                 for j in 1:length(w)  # j=1
#                     nw[l][j] = (abs(w[j]) * mask[l][j]) .^ p 
#                 end
#             end

#             # calc γ in and out
#             for l in 1:depth # l = 3
#                 A_mul_B!(γin[l+1], nw[l], γin[l])
#                 il = depth-l+1
#                 A_mul_B!(γout[il], γout[il+1], nw[il])
#             end

#             if p != 2.
#                 for l in 1:depth+1
#                     for j in 1:length(γin[l])
#                         γin[l][j]  = γin[l][j]  .^ (2/p)
#                         γout[l][j] = γout[l][j] .^ (2/p)
#                     end
#                 end
#             end

#             # calc final normalization matrices, store in nw
#             for l in 1:depth # l = 1
#                 A_mul_B!(nw[l], γin[l+1], γout[l])
#             end



#         δ    = out[1] - res[i]
#         λ    = - λ0 * δ * pond[i]

#         # # clamping of big gradients
#         # l2norm = sqrt(sum( map(sumabs2, out[2:end-1] ) ) )
#         # λ    = λ * (l2norm > 1. ? 1. / l2norm : 1.)

#         # # calc update
#         # for l in 1:length(dls) # l = 1
#         #     for j in 1:length(dls[l])
#         #         dls[l][j] = μ * dls[l][j] +
#         #                     (1-μ) * λ / nw[l][j] * mask[l][j] * 
#         #                     out[l+1][j]
#         #     end
#         # end

#         # alt version
#         l2nw = 0.
#         l2grad = 0.
#         for l in 1:length(dls) # l = 1
#             l2nw   += sumabs2(nw[l])
#             l2grad += sumabs2(out[l+1])
#         end
#         λ = λ * sqrt(l2nw) / (l2grad > 1. ? sqrt(l2grad) : 1.)

#         # calc update
#         for l in 1:length(dls) # l = 1
#             for j in 1:length(dls[l])
#                 if nw[l][j] > 1e-6
#                     dls[l][j] = μ * dls[l][j] +
#                                 (1-μ) * λ / nw[l][j] * mask[l][j] * 
#                                 out[l+1][j]
#                 else
#                     dls[l][j] = μ * dls[l][j]
#                 end
#             end
#         end

#         # apply update
#         for l in 1:length(dls) # l = 1
#             for j in 1:length(dls[l])
#                 ls[l][j] += dls[l][j]
#             end
#         end

#         if freq > 0 && (k % freq == 0)
#             m, s = score(f, ls, subset, vars, res, pond)
#             println("k : $k λ : $(round(λ0,4)) mean : $(round(m,5)) stderr : $(round(s,5))")
#         end

#         ri = ri == length(subset) ? 1 : ri+1
#     end

#     score(f, ls, subset, vars, res, pond)
# end

