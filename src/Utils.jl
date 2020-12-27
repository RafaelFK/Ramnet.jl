module Utils

export stack, ambiguity

"""
    check_length_consistency(inputs::AbstractVector...)

Raise a `DimensionMismatch` exception if the input vectors are not all the same
length. Does nothing otherwise.

# Examples

```jldoctest
julia> check_length_consistency([1,2,3], [4,5])
ERROR: DimensionMismatch("Input vectors should have the same length!")
[...]
```
"""
function check_length_consistency(inputs::AbstractVector...)
    if any(!=(length(first(inputs))), map(length, inputs))
        throw(DimensionMismatch("Input vectors should have the same length!"))
    end
end

"""
    check_length_consistency(inputs::AbstractMatrix...)

Raise a `DimensionMismatch` exception if the input matrices don't have the same
number of columns. Does nothing otherwise.

# Examples

```jldoctest
julia> check_length_consistency([1 2 3; 4 5 6], [7 8; 9 10])
ERROR: DimensionMismatch("Input vectors should have the same length!")
[...]
```
"""
function check_length_consistency(inputs::AbstractMatrix...)
    if any(!=(size(first(inputs))[2]), map(i -> size(i)[2], inputs))
        throw(DimensionMismatch("input matrices should have the same number of columns"))
    end
end

# TODO: Consider renaming it to stack_rows
"""
    stack(inputs::AbstractVector...)

Stack the input vectors as rows of a matrix.

# Examples
```jldoctest
julia> stack([1,2,3], [4,5,6])
2×3 Array{Int64,2}:
 1  2  3
 4  5  6
```
"""
function stack(inputs::AbstractVector...)
    check_length_consistency(inputs...)

    mapreduce(permutedims, vcat, inputs)
end

"""
    stack(inputs::AbstractMatrix...)

Stack the input matrices vertically.
"""
function stack(inputs::AbstractMatrix...)
    check_length_consistency(inputs...)

    reduce(vcat, inputs)
end

# TODO: Untested
function accuracy(y_predict::AbstractVector{T}, y_actual::AbstractVector{T}) where {T}
    check_length_consistency(y_predict, y_actual)

    sum(y_predict .== y_actual) / length(y_predict)
end

"""
    hasdraw(X::Tuple)

Return `true` if there is draw in the tuple `X` and `false` otherwise. A draw 
occurs when the largest value in the tuple is non-unique.
"""
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

# TODO: Add example to docstring
"""
    ambiguity(responses...)

Measure the ambiguity in a multidiscriminator classifier prediction.

The ambiguity is the average number of ties in the responses of the
discriminators that compose a multidiscriminator classifier when predicting the
classes of a test set.
"""
function ambiguity(responses::AbstractVector{T}...) where {T <: Integer}
    check_length_consistency(responses...)
    count = 0
    
    for res in Iterators.zip(responses...)
        if hasdraw(res)
            count += 1
        end
    end

    return count / (length ∘ first)(responses)
end

end