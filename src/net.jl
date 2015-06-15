import Base.size

immutable NN{n}
    ws::NTuple{n, Matrix{Float64}}
    ns::NTuple{n, NodeFunc}
    ls::NTuple{Int}

    function NN(ws::NTuple{n, Matrix{Float64}}, 
                ns::NTuple{n, NodeFunc})

        for i in 1:length(ws)-1
            @assert size(ws[i],1) == size(ws[i+1],2) "size of levels $i and $(i+1) do not fit"
        end
        
        ls = tuple([ size(w,1) for w in ws ]...)
        new(ws, ns, ls)
    end
end   

NN{n}(ws::NTuple{n, Matrix{Float64}}, ns::NTuple{n, NodeFunc}) = NN{n}(ws, ns)

size(nn::NN) = nn.ls
inputsize(nn::NN) = size(nn.ws[1],2)
depth{n}(nn::NN{n}) = n
