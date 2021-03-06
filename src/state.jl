

type TrainState{n,m}
    ss::NTuple{n, Array{Float64}}   # holds the layers' output
    as::NTuple{n, Array{Float64}}   # holds the activation
    ds::NTuple{n, Array{Float64}}   # temporary storage for gradient
    dws::NTuple{n, Matrix{Float64}} # holds the gradient of weight matrices
    dbs::NTuple{n, Vector{Float64}} # holds the gradient of bias vector
    ls::NTuple{Int}
end   

# creates a TrainState adapted to a given Neural net and # of samples
function TrainState(nn::NN, nsamples::Int) # nsamples = 3
    siz = size(nn)
    ss  = tuple([Array(Float64, s, nsamples) for s in siz]...)
    # dws = tuple( Array(Float64, siz[1], inputsize(nn)),
    #             [Array(Float64, siz[i], siz[i-1]) for i in 2:length(siz)]...)
    dws = tuple( map(similar, nn.ws)... )
    dbs = tuple( map(similar, nn.bs)... )

    TrainState{depth(nn),nsamples}(ss, 
                                   map(similar,ss), 
                                   map(similar,ss), 
                                   dws, 
                                   dbs, 
                                   siz)
end    

compatible(nn::NN, ss::TrainState) = size(nn) == size(ss)

size(ss::TrainState) = ss.ls
nsamples{n,m}(ss::TrainState{n,m}) = m

function cycle{T<:FloatingPoint}(nn::NN, 
    input::StridedArray{T},
    output::StridedArray{T},
    metric::PreMetric,
    weights::StridedVector{T}) 

    cycle!(TrainState(nn, size(input,2)), nn, input, output, metric, weights)
end

function cycle!{n,m,T<:FloatingPoint}(s::TrainState{n,m}, 
    net::NN{n}, 
    input::StridedArray{T},
    output::StridedArray{T},
    metric::PreMetric,
    weights::StridedVector{T})

    @assert compatible(net,s) "incompatible network and state"
    @assert size(input,1) == inputsize(net) "incompatible size of input and network"
    @assert size(output,1) == size(net.ws[end],1) "incompatible size of output and network"
    @assert size(output,2) == size(input,2) "incompatible # samples between input and output"
    if ndims(input) == 1
        @assert m == 1 "incompatible # of samples in input"
    else 
        @assert m == size(input,2) "incompatible # of samples in input"
    end

    # forward pass
    prev = input
    for i in 1:depth(net)
        A_mul_B!(s.as[i], net.ws[i], prev)
        # bias term
        for j in 1:size(s.as[i],1)
            bias = net.bs[i][j]
            for k in 1:size(s.as[i],2)
                s.as[i][j,k] += bias
            end
        end
        nfunc!(s.ss[i], s.as[i], net.ns[i])
        prev = s.ss[i]
    end

    # backward pass
    dmetric!(s.ds[end], s.ss[end], output, metric, weights)
    for i in depth(net):-1:2
        dnfunc!(s.ds[i], s.as[i], s.ss[i], net.ns[i])
        sum!(s.dbs[i], s.ds[i])                    # gradient on bias term
        A_mul_Bt!( s.dws[i],   s.ds[i], s.ss[i-1]) # gradient on weight matrix
        At_mul_B!(s.ds[i-1], net.ws[i], s.ds[i])
    end
    dnfunc!(s.ds[1], s.as[1], s.ss[1], net.ns[1])
    sum!(s.dbs[1], s.ds[1])                        # gradient on bias term
    A_mul_Bt!( s.dws[1],   s.ds[1], input)

    s
end

function calc{T<:FloatingPoint}(net::NN, input::StridedArray{T})
    @assert size(input,1) == inputsize(net) "incompatible size of input and network"

    # forward pass
    prev = input
    for i in 1:depth(net)
        prev = net.ws[i] * prev
        # bias term
        for j in 1:size(prev,1)
            bias = net.bs[i][j]
            for k in 1:size(prev,2)
                prev[j,k] += bias
            end
        end
        nfunc!(prev, prev, net.ns[i])
    end

    prev
end


