

type DataSet{m}
    X::Matrix{Float64}        # n x m matrix of n input variables by m samples
    Y::Matrix{Float64}        # k x m matrix of k output variables by m samples
    metric::PreMetric         # error measurement method (from Distances package)
    weights::Vector{Float64}  # samples weighting

    function DataSet(X::Matrix{Float64},
        Y::Matrix{Float64},
        metric::PreMetric,
        weights::Vector{Float64})

        @assert size(X,2) == size(Y,2) "incompatible number of samples between input and output"
        @assert size(X,2) == length(weights) "incompatible number of samples and weights"

        new(X, Y, metric, weights)
    end
end


DataSet(X::Matrix{Float64},
        Y::Matrix{Float64},
        metric::PreMetric=SqEuclidean(),
        weights::Vector{Float64}=ones(size(X,2))) = 
    DataSet{size(X,2)}(X, Y, metric, weights)

# if vector, assume the unique dimension is along samples
function DataSet(X::Union(Vector{Float64}, Float64), 
                 Y::Union(Vector{Float64}, Float64), 
                 metric::PreMetric=SqEuclidean(),
                 weights::Vector{Float64}=length(X))

    DataSet{1}(reshape(X, 1, length(X)), 
               reshape(Y, 1, length(Y)), 
               metric, weights)
end

compatible(nn::NN, ts::DataSet) = (inputsize(nn)==size(ts.X,1)) && (size(nn)[end]==size(ts.Y,1))
nsamples{m}(ts::DataSet{m}) = m

function shuffle!(ts::DataSet; seed::Integer=0)
    (seed != 0) && srand(seed)
    idx = shuffle(collect(1:nsamples(ts)))

    ts.X       = ts.X[:,idx]
    ts.Y       = ts.Y[:,idx]
    ts.weights = ts.weights[idx]
end

function split(ts::DataSet, fraction=0.8)
    pos = round(Int, nsamples(ts) * fraction)

    ( DataSet(ts.X[:,1:pos-1],
              ts.Y[:,1:pos-1],
              ts.metric,
              ts.weights[1:pos-1]), 
      DataSet(ts.X[:,pos:end],
              ts.Y[:,pos:end],
              ts.metric,
              ts.weights[pos:end]) )
end


function show{m}(io::IO, ts::DataSet{m})
    print(io, "training data : ")
    print(io, "[$m] samples, ")
    print(io, "input size [$(size(ts.X,1))], ")
    print(io, "output size [$(size(ts.Y,1))], ")
    print(io, "metric [$(typeof(ts.metric))]")
    any(ts.weights .!= 1.) && print(io, ", with weights")
    println(io,"")
end
