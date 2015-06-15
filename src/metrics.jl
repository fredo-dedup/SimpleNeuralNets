

function dmetric!(dest, x, y, ::SqEuclidean)
    for i in 1:length(x)
        dest[i] = 2*(x[i]-y[i])
    end
end

