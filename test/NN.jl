
    nnex = quote
        X1 = W1 * X0
        X1 = log(1. + exp(X1))
        X2 = W2 * X1
        X2 = (1. .- exp(-X2)) ./ (1. + exp(-X2))
        sum((X2 - Y) .* (X2 - Y))
    end


    nn = S.NN((randn(10,20), randn(5,10)), (S.Relu(), S.Relu()))
    size(nn)
    S.depth(nn)

    s0 = S.calc(nn, rand(20))
    copy!(s0.ss[1], ones(20))
    S.compatible(nn, s0)
    S.calc!(nn, s0)

    s0 = S.calc(nn, rand(20,10))
    copy!(s0.ss[1], ones(20))
    S.compatible(nn, s0)
    S.calc!(nn, s0)

    a = (1,2,3)
    typeof(a)
    a[1]
    a[1] = 2


    nn = S.NN((randn(1,1),), (S.Relu(),))
    S.calc(nn, [1.])

    nn = S.NN((randn(1,1),), (S.SoftRelu(),))
    S.calc(nn, [1.])



a = (45, 12)
typeof(a)
isa(a, NTuple{2, Float64})


Pkg.generate("SimpleNeuralNets","MIT")
Pkg.status()
