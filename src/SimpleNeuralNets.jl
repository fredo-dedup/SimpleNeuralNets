module SimpleNeuralNets

    using Reexport
    @reexport using Distances
    using ArrayViews
    
    import Base: size
    export NN, 
           size, depth, inputsize, nsamples,
           cycle, cycle!, calc,
           Sigmoid, Tanh, Relu, SoftRelu,
           sgd


    include("nodefunc.jl")
    include("net.jl")
    include("state.jl")
    include("metrics.jl")
    include("solvers.jl")

end # module
