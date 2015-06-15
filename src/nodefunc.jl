
abstract NodeFunc

type Relu <: NodeFunc ; end
nfunc(a, ::Relu)     = max(0., a)
dnfunc(ds, a, s, ::Relu) = a < 0. ? 0. : ds

type SoftRelu <: NodeFunc ; end
nfunc(a, ::SoftRelu) = log(1. + exp(a))
dnfunc(ds, a, s, ::SoftRelu) = exp(a) / exp(s) * ds

type Sigmoid <: NodeFunc ; end
nfunc(a, ::Sigmoid) = 1. / (1. + exp(-a))
dnfunc(ds, a, s, ::Sigmoid) = exp(-a) / (s*s) * ds

type Tanh <: NodeFunc ; end
nfunc(a, ::Tanh) = (1. - exp(-2a)) / (1. + exp(-2a))
dnfunc(ds, a, s, ::Tanh) = (1 - s*s) * ds

# exp(x) / (1+exp(x))

function nfunc!(s::AbstractArray, a::AbstractArray, f::NodeFunc)
    for i in 1:length(s)
        s[i] = nfunc(a[i], f)
    end
end

function dnfunc!(ds::AbstractArray, a::AbstractArray, s::AbstractArray, f::NodeFunc)
    for i in 1:length(s)
        ds[i] = dnfunc(ds[i], a[i], s[i], f)
    end
end