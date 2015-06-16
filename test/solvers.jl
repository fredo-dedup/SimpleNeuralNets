reload("SimpleNeuralNets"); S=SimpleNeuralNets


X = vars'
Y = res'
# subset=idx; minibatch=100; steps=100; freq=0; noise=0.

nn = S.NN([19,10,10,1], [S.Relu(), S.Sigmoid()])
# copy!(nn.ws[1], ls[1])
# copy!(nn.ws[2], ls[2])
# copy!(nn.ws[3], ls[3])
λ0, μ = 1e-2, 0.99
S.sgd(nn, λ0, μ, X, Y, subset=idx, freq=200, steps=1000)

##############
reload("SimpleNeuralNets"); S=SimpleNeuralNets
nn = S.NN([19,10,10,1], [S.Relu(), S.Sigmoid()])

@time S.calc(nn, X) # 3.776
@time nn.ws[1] * X # 1.48 s

tst = convert(Array{Float64}, nn.ws[1])
@time tst * X # 0.005 s


@time S.calc2(nn, X) # 3.776
Profile.clear()
@profile S.calc(nn,X)
Profile.print()

Profile.print()
@profile S.sgd(nn, λ0, μ, X, Y, subset=idx, freq=200, steps=100)
Profile.print()


isa(X, StridedArray)
methods(S.cycle!, (S.TrainState, S.NN, SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},2}, SubArray{Float64}))
methods(S.cycle!, (S.TrainState, S.NN, StridedArray{Float64}, StridedArray))

methods(S.cycle!, (SimpleNeuralNets.TrainState, SimpleNeuralNets.NN, SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},1}, SubArray))


typeof(sub(X,:,1:100))
cycle!(::SimpleNeuralNets.TrainState{3,100}, ::SimpleNeuralNets.NN{3}, ::SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},1}, ::SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},1})

T = sub(vars,120:320,:)
isa(T, SubArray)
isa(T, StridedArray)
methods(A_mul_B!, (DenseArray, SubArray, DenseArray))

@time S.sgd(nn, λ0, μ, X, Y, subset=idx, steps=10000) # 1.4 680Mb
@time S.sgd(nn, λ0, μ, X, Y, minibatch=10, subset=idx, steps=100000) # 10s 5400Mb
@time S.sgd(nn, λ0, μ, X, Y, minibatch=1000, subset=idx, steps=1000) # 0.52s 210Mb

# nn = S.NN((randn(10,19), randn(10,10), randn(1,10)), (S.Relu(), S.Relu(), S.Relu()))
# nn = S.NN((ls...,), (S.Relu(), S.Relu(), S.Relu()))

# sgd(nn, λ0, μ, X, Y, subset=idx, freq=200, steps=10000)

# @time sgd(nn, λ0, μ, X, Y, freq=200, steps=1000)
# @time sgd(nn, λ0, μ, X, Y, freq=0, steps=1000)

