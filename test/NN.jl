

    A = [1 2 ; 3 4]
    B = [10, 10]
    sum!(B, A)
    sum!(A, B)


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

    nn = S.NN([10,10,10,10,1], [ S.Tanh(), S.Tanh()] )
    ts = S.TrainState(nn, 10)

    size(nn)
    size(ts)

    ts2 = S.cycle(nn, ones(20,10), ones(5,10))
    S.cycle!(ts, nn, ones(10,10), ones(1,10))

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

########### ANN3 #################################"

    import Bokeh
    # using ReverseDiffSource
    reload("SimpleNeuralNets"); s = SimpleNeuralNets

    f(x) = exp(-(x-1)^2/5)*cos(x)*sign(x)

    xs = collect(-5:0.001:5)
    X = Float64[ x for x in xs ]'
    Y = Float64[ f(x) for x in xs ]'

    nn = s.NN([1,10,10,10,1], [ s.Relu(), s.Tanh()] )

    ts = s.TrainingSet(X, Y)
    any(ts.weights .!= 1.0)

    λ0, μ = 1e-3, 0.9
    s.sgd(nn, ts, λ0, μ, freq=200, steps=1000)

    ts2 = s.TrainingSet(X, Y, SqEuclidean(), rand(size(X,2)))
    s.sgd(nn, ts2, λ0, μ, freq=200, steps=1000)

    s.colwise(a,b,c)
    length(a.weights)

    px = s.calc(nn, X)
    sum(s.colwise(ts2.metric, px, ts2.Y))


    λ0, μ = 1e-3, 0.9
    s.sgd(nn, λ0, μ, X, Y, freq=200, steps=10000)
    λ0, μ = 1e-5, 0.999
    s.sgd(nn, λ0, μ, X, Y, freq=200, steps=10000)

    Bokeh.hold(true)
    Bokeh.plot(f, -5:5, "k-")
    px = collect(-5:0.01:5)
    py = s.calc(nn, px')'
    Bokeh.plot(px, py, "b-")
    Bokeh.showplot("/tmp/plot.html")
    Bokeh.hold(false)

    nn = s.NN([1,10,10,10,1], [ s.SoftRelu(), s.Tanh()] )
    λ0, μ = 1e-3, 0.9
    s.sgd(nn, λ0, μ, X, Y, freq=200, steps=10000)
    λ0, μ = 1e-3, 0.
    s.sgd(nn, λ0, μ, X, Y, freq=1000, steps=10000)

    Bokeh.hold(true)
    Bokeh.plot(f, -5:5, "k-")
    px = collect(-4:0.1:4)
    py = s.calc(nn, px')'
    Bokeh.plot(px, py, "b-")
    Bokeh.showplot("/tmp/plot.html")
    Bokeh.hold(false)