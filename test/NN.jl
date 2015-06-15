
    using ReverseDiffSource

        Nn = [20,10,1]
        function ann(l1, l2, x1)
            x2 = l1 * x1 
            x2 = max(0., x2) #relu
            x3 = l2 * x2 
            x3 = (max(0., x3[1]))^2 #relu
        end

        # ls = Array[ nn.ws...]
        ls = init(Nn)
        dann = rdiff(ann, (ls..., ones(20) ))
        out = dann(ls..., ones(20) )

    nn = S.NN((ls[1], ls[2]), (S.Relu(), S.Relu()))
    ts2 = S.cycle(nn, ones(20), zeros(1))

        out[2] 
        ts2.dws[1]

        out[3] 
        ts2.dws[2]

        ts2.ds[end]

        out[3] ./ ts2.dws[2]

    r1 = S.calc(nn, ones(20))[1] # 1.16217
    r2 = ann(ls..., ones(20))
    r1-r2

    ls[1][9,1] += 1e-10
    (ann(ls..., ones(20)) - r1) * 1e10  # -0.8600

    nn.ws[1][9,1] += 1e-10
    (S.calc(nn, ones(20)) - r2) * 1e10  # -0.8600


    reload("SimpleNeuralNets"); S=SimpleNeuralNets

    nn = S.NN((randn(10,20), randn(5,10)), (S.Relu(), S.Relu()))
    ts = S.TrainState(nn, 10)

    size(nn)
    size(ts)

    ts2 = S.cycle(nn, ones(20,10), ones(5,10))
    S.cycle!(ts, nn, ones(20,10), ones(5,10))

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
