

function dmetric!(dest, x, y, ::SqEuclidean, weights)
    for j in 1:size(x,2)
        w = weights[j]
        for i in 1:size(x,1)
            dest[i,j] = 2 * (x[i,j]-y[i,j]) * w
        end
    end
end

