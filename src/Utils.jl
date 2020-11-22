module Utils

export stack

function stack(inputs::AbstractVector...)
    if any(!=(length(first(inputs))), map(length, inputs))
        throw(DimensionMismatch("Input vectors should have the same length!"))
    end

    mapreduce(permutedims, vcat, inputs)
end

function stack(inputs::AbstractMatrix...)
    if any(!=(size(first(inputs))[2]), map(i -> size(i)[2], inputs))
        throw(DimensionMismatch("Input vectors should have the same length!"))
    end

    reduce(vcat, inputs)
end

end