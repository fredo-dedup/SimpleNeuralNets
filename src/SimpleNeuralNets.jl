module SimpleNeuralNets

    using Reexport
    @reexport using Distances


    import Base: size
    export size, calc, cycle, cycle!

    include("nodefunc.jl")
    include("net.jl")
    include("state.jl")
    include("metrics.jl")

end # module
