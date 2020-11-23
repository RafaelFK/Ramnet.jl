module Utils

export stack, ambiguity

# TODO: Abstract away the length check
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

# TODO: Vectors must have the same length
# TODO: Untested
function accuracy(y_predict::AbstractVector{T}, y_actual::AbstractVector{T}) where {T}
    sum(y_pred .== y_target) / length(y_pred)
end

function hasdraw(X::Tuple)
    length(X) == 0 && throw(ArgumentError("X must have at least one element."))

    repeats = 0
    largest = first(X)

    for x in X
        if x > largest
            largest = x
            repeats = 1
        elseif x == largest
            repeats += 1
        end
    end

    return repeats != 1
end

# TODO: Vectors must have the same length
function ambiguity(responses::AbstractVector{T}...) where {T <: Integer}
    count = 0
    
    for res in Iterators.zip(responses...)
        if hasdraw(res)
            count += 1
        end
    end

    return count / (length ∘ first)(responses)
end

# function ambiguity(model, X::T) where {T <: AbstractMatrix{Bool}}
#     ambiguity((values ∘ predict_response)(model, X)...)
# end

end