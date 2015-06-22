module SimpleNeuralNets

    using Reexport
    @reexport using Distances
    
    import Base: size, show, split, shuffle! #, slice
    export NN, DataSet,
           size, depth, inputsize, nsamples,
           split, shuffle!,
           calc,
           Sigmoid, Tanh, Relu, SoftRelu,
           sgd


    include("nodefunc.jl")
    include("net.jl")
    include("state.jl")
    include("metrics.jl")
    include("dataset.jl")
    include("solvers.jl")

end # module
