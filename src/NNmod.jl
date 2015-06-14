module S
    import Base.size

    abstract NodeFunc

    type Relu <: NodeFunc ; end
    nfunc(x, ::Relu) = max(0., x)

    type SoftRelu <: NodeFunc ; end
    nfunc(x, ::SoftRelu) = log(1. + exp(x))

    function nfunc!(s::Array, f::NodeFunc)
        for i in 1:length(s)
            s[i] = nfunc(s[i],f)
        end
    end


    immutable NN{n}
        ws::NTuple{n, Matrix{Float64}}
        dws::NTuple{n, Matrix{Float64}} # holds the gradient
        ns::NTuple{n, NodeFunc}
        ls::NTuple{Int}

        function NN(ws::NTuple{n, Matrix{Float64}}, 
                    ns::NTuple{n, NodeFunc})

            for i in 1:length(ws)-1
                @assert size(ws[i],1) == size(ws[i+1],2) "size of levels $i and $(i+1) do not fit"
            end
            
            ls = tuple([ size(w,1) for w in ws ]...)
            new(ws, map(similar, ws), ns, ls)
        end
    end   

    NN{n}(ws::NTuple{n, Matrix{Float64}}, ns::NTuple{n, NodeFunc}) = NN{n}(ws, ns)

    size(nn::NN) = nn.ls
    inputsize(nn::NN) = size(nn.ws[1],2)
    depth{n}(nn::NN{n}) = n


    type NNState{n,m}
        ss::NTuple{n, Array{Float64}}
        as::NTuple{n, Array{Float64}} # holds the activation (useful for gradient calc)
        ls::NTuple{Int}

        function NNState(ss::NTuple{n, Array{Float64}})
            if ndims(ss[1]) == 1  # single sample
                @assert all([map(ndims, ss)...] .== 1) "incompatible # samples"
                ls = tuple([ size(s,1) for s in ss]...)
                new(ss, map(similar,ss), ls)
            elseif ndims(ss[1]) == 2  # multiple samples
                @assert all([map(ndims, ss)...] .== 2) "incompatible # samples"
                nsa = size(ss[1],2)
                @assert all(map(x->size(x,2), ss) .== nsa) "incompatible # samples"
                ls = tuple([ size(s,1) for s in ss]...)
                new(ss, map(similar,ss), ls)
            end
        end
    end   

    NNState{n}(ss::NTuple{n, Vector{Float64}}) = NNState{n, 1}(ss)
    NNState{n}(ss::NTuple{n, Matrix{Float64}}) = NNState{n, size(ss[1],2)}(ss)

    compatible(nn::NN, ss::NNState) = size(nn) == size(ss)

    size(ss::NNState) = ss.ls
    nsamples{n,m}(ss::NNState{n,m}) = m

    function forward(nn::NN, input::AbstractArray{Real}) # input = rand(20)
        if ndims(input) == 1
            ss = tuple([ Array(Float64, sz) for sz in size(nn)[2:end]]...)
        else 
            nsa = size(input,2)
            ss = tuple([ Array(Float64, sz, nsa) for sz in size(nn)[2:end]]...)
        end
        calc!(NNState(ss), nn, input)
    end

    function forward!{n,m}(ss::NNState{n,m}, nn::NN{n}, input::AbstractArray{float64})
        @assert compatible(nn,ss) "incompatible network and state"
        @assert size(input,1) == inputsize(nn) "incompatible size of input and network"
        if ndims(input) == 1
            @assert m == 1 "incompatible # of samples in input"
        else 
            @assert m == size(input,2) "incompatible # of samples in input"
        end

        prev = input
        for i in 1:depth(nn)
            A_mul_B!(ss.ss[i], nn.ws[i], prev)
            prev = ss.ss[i]
            nfunc!(prev, nn.ns[i])
        end

        ss
    end

    function backward!{n,m}(nn2::NN{n}, ss::NNState{n,m}, nn::NN{n})
        @assert compatible(nn,ss) "incompatible network and state"
        @assert compatible(nn2,ss) "incompatible network and state"
        @assert size(input,1) == inputsize(nn) "incompatible size of input and network"
        @assert size(input,1) == inputsize(nn2) "incompatible size of input and network"
        if ndims(input) == 1
            @assert m == 1 "incompatible # of samples in input"
        else 
            @assert m == size(input,2) "incompatible # of samples in input"
        end

        prev = ones(ss.ss[end])
        for i in depth(nn):-1:1
            bnfunc!(prev, ss[i], nn.ns[i])
            A_mul_Bt(nn2[i], prev, nn.ws[i])
            prev = ss.ss[i]
            nfunc!(prev, nn.ns[i])
        end

        ss
    end





end