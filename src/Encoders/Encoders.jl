module Encoders

export encode

abstract type AbstractEncoder end

function encode! end
function encode end

# TODO: Consider making 'encode!' the "Internal usage" one
# Internal usage function
# function _encode! end

########################### Generic encoding methods ###########################
function encode(encoder::E, x::T) where {E <: AbstractEncoder,T <: Real}
    pattern = Vector{Bool}(undef, encoder.resolution)

    encode!(encoder, x, pattern)
end

function encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, x, pattern)

    if encoder.min isa AbstractVector
        for (i, col) in Iterators.enumerate(eachcol(pattern))
            encode!(encoder, encoder.min[i], encoder.max[i], x[i], col)
        end
    else
        for (i, col) in Iterators.enumerate(eachcol(pattern))
            encode!(encoder, x[i], col)
        end
    end

    pattern
end

# Flat version. All values in the input vector are encoded and placed in a single output vector
function encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, x, pattern)

    if encoder.min isa AbstractVector
        for (i, slice) in Iterators.enumerate(Iterators.partition(pattern, encoder.resolution))
            encode!(encoder, encoder.min[i], encoder.max[i], x[i], slice)
        end
    else
        for (i, slice) in Iterators.enumerate(Iterators.partition(pattern, encoder.resolution))
            encode!(encoder, x[i], slice)
        end
    end

    pattern
end

function encode(encoder::E, x::AbstractVector{T}; flat=true) where {E <: AbstractEncoder,T <: Real}
    if flat
        pattern = Vector{Bool}(undef, encoder.resolution * length(x))
    else
        pattern = Matrix{Bool}(undef, encoder.resolution, length(x))
    end

    encode!(encoder, x, pattern)
end

function encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, X, pattern)

    for (slice, row) in Iterators.zip(eachslice(pattern; dims=3), eachrow(X))
        encode!(encoder, row, slice)
    end

    pattern
end

# Flat version. All values in each row of the input matrix are encoded and placed in a single row of the output matrix
function encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, X, pattern)

    for (slice, row) in Iterators.zip(eachrow(pattern), eachrow(X))
        encode!(encoder, row, slice)
    end

    pattern
end

function encode(encoder::E, X::AbstractMatrix{T}; flat=true) where {E <: AbstractEncoder,T <: Real}
    if flat
        pattern = Matrix{Bool}(undef, size(X, 1), encoder.resolution * size(X, 2))
    else
        pattern = Array{Bool,3}(undef, encoder.resolution, size(X, 2), size(X, 1))
    end

    encode!(encoder, X, pattern)
end

########################## Generic validation methods ##########################
function validate_input_output(encoder::E, x::T, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    # If more then one minimum/maximum were especified, it doesn't make sense to try to encode a scalar
    length(encoder.min) != 1 && throw(DomainError(x, "expected vector of length $(length(encoder.min))"))

    length(pattern) != encoder.resolution && throw(
      DimensionMismatch(
        "output pattern must have the length of the thermometer's resolution"))

    nothing
end

function validate_input_output(encoder::E, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
      DimensionMismatch(
        "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != length(x) && throw(
      DimensionMismatch(
          "the number of columns of the output pattern must equal x's length"))

    encoder.min isa AbstractVector && length(x) != length(encoder.min) && throw(
      DimensionMismatch(
        "the length of x must match the lengths of the min and max vectors"))

    nothing
end

function validate_input_output(encoder::E, x::AbstractVector{T}, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    length(pattern) != length(x) * encoder.resolution && throw(
      DimensionMismatch(
        "the length of the output pattern must equal the thermometer's resolution times the length of x"))

    encoder.min isa AbstractVector && length(x) != length(encoder.min) && throw(
      DimensionMismatch(
        "the length of x must match the lengths of the min and max vectors"))
    nothing
end

function validate_input_output(encoder::E, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
      DimensionMismatch(
        "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != size(X, 2) && throw(
      DimensionMismatch(
          "the number of columns of the output pattern must equal X's number of columns"))

    size(pattern, 3) != size(X, 1) && throw(
      DimensionMismatch(
          "the depth of the output pattern must equal X's number of rows"))

    encoder.min isa AbstractVector && size(X, 2) != length(encoder.min) && throw(
      DimensionMismatch(
        "the number of columns of X must match the lengths of the min and max vectors"))

    nothing
end

function validate_input_output(encoder::E, X::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != size(X, 1) && throw(
        DimensionMismatch(
          "the number of rows of the output pattern must equal the the number of rows of X"))
    
    size(pattern, 2) != size(X, 2) * encoder.resolution && throw(
          DimensionMismatch(
              "the number of columns of the output pattern must equal the thermometer's resolution times X's number of columns"))

    encoder.min isa AbstractVector && size(X, 2) != length(encoder.min) && throw(
      DimensionMismatch(
        "the number of columns of X must match the lengths of the min and max vectors"))

    nothing
end

include("Thermometer.jl")
export Thermometer

include("CircularThermometer.jl")
export CircularThermometer

end