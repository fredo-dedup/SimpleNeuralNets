import Base.size

immutable NN{n}
    ws::NTuple{n, Matrix{Float64}}   # weight matrices
    bs::NTuple{n, Vector{Float64}}   # bias vectors
    ns::NTuple{n, NodeFunc}          # node function (sigmoid, tanh, etc.)
    ls::NTuple{Int}                  # size vector

    function NN(ws::NTuple{n, Matrix{Float64}}, 
                bs::NTuple{n, Vector{Float64}},
                ns::NTuple{n, NodeFunc})

        for i in 1:length(ws)-1
            @assert size(ws[i],1) == size(ws[i+1],2) "size of levels $i and $(i+1) do not fit"
        end

        for i in 1:length(ws)
            @assert size(ws[i],1) == size(bs[i],1)   "size of bias and weight matrix do not fit at level $i"
        end
        
        ls = tuple([ size(w,1) for w in ws ]...)
        new(ws, bs, ns, ls)
    end
end   

# NN{n}(ws::NTuple{n, Matrix{FloatingPoint}}, ns::NTuple{n, NodeFunc}) = NN{n}(ws, ns)

size(nn::NN) = nn.ls
inputsize(nn::NN) = size(nn.ws[1],2)
depth{n}(nn::NN{n}) = n


function NN(sz::Vector{Int}, nf=Relu(); seed=0)
    @assert length(sz) > 1 "Net needs at least 2 layers"
    ws = Matrix{Float64}[]
    bs = Vector{Float64}[]
    srand(seed)
    for i in 1:length(sz)-1
        push!(ws, randn(sz[i+1], sz[i]) .* sqrt(6/(sz[i+1]+sz[i])))
        push!(bs, zeros(sz[i+1]))
    end

    if isa(nf, NodeFunc)
        ns = NodeFunc[ nf for i in 1:length(ws)]
    elseif isa(nf, Vector) && length(nf)==1
        ns = NodeFunc[ nf[1] for i in 1:length(ws)]
    elseif isa(nf, Vector) && length(nf)==2
        ns = NodeFunc[ [ nf[1] for i in 1:length(ws)-1] ; nf[2] ]
    elseif isa(nf, Vector) && length(nf)==length(ws)
        ns = NodeFunc[ nf[i] for i in 1:length(ws) ]
    else
        error("net depth and node function length do not match")
    end

    NN{length(ws)}(tuple(ws...), tuple(bs...), tuple(ns...))
end


function show(io::IO, nn::NN)
    print(io, "neural net : ")
    print(io, "layers sizes $( Int[ inputsize(nn) , size(nn)... ]), ")
    print(io, "node functions $( map(typeof, nn.ns) )")
    println(io,"")
end