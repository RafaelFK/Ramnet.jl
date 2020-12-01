module Encoders

export Thermometer
export encode!, encode

abstract type AbstractEncoder end

function encode! end
function encode end

# TODO: Consider using a parametric type for min and max' types and having it be
#       infered from the promotion of the min and max arguments. After that, the
#       type of the encoded value could be constrained to be a subtype of this
#       parameter
struct Thermometer <: AbstractEncoder
    min::Float64
    max::Float64
    resolution::Int # TODO: This could be unsigned

    function Thermometer(min::Float64, max::Float64, resolution::Int)
        max <= min && throw(ArgumentError(
          "the maximum of the thermometer may not be lesser or equal to its minimum"
        ))

        resolution < 1 && throw(DomainError(
          resolution,
          "the thermometer's resolution must be greater then zero"
        ))

        new(min, max, resolution)
    end
end

function encode!(encoder::Thermometer, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    length(pattern) != encoder.resolution && throw(
      DimensionMismatch(
        "output pattern must have the length of the thermometer's resolution"))
    
    for i in eachindex(pattern)
        pattern[i] = x > encoder.min + (i - 1) * (encoder.max - encoder.min) / encoder.resolution
    end

    pattern
end

function encode(encoder::Thermometer, x::T) where {T <: Real}
    pattern = Vector{Bool}(undef, encoder.resolution)

    encode!(encoder, x, pattern)
end

function encode!(encoder::Thermometer, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
        DimensionMismatch(
          "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != length(x) && throw(
        DimensionMismatch(
            "the number of columns of the output pattern must equal x's length"))

    for (i, col) in Iterators.enumerate(eachcol(pattern))
        encode!(encoder, x[i], col)
    end

    pattern
end

function encode(encoder::Thermometer, x::AbstractVector{T}) where {T <: Real}
    pattern = Matrix{Bool}(undef, encoder.resolution, length(x))

    encode!(encoder, x, pattern)
end

function encode!(encoder::Thermometer, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
      DimensionMismatch(
        "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != size(X, 2) && throw(
        DimensionMismatch(
            "the number of columns of the output pattern must equal X's number of columns"))

    size(pattern, 3) != size(X, 1) && throw(
      DimensionMismatch(
          "the depth of the output pattern must equal X's number of rows"))

    for (slice, row) in Iterators.zip(eachslice(pattern; dims=3), eachrow(X))
        encode!(encoder, row, slice)
    end

    pattern
end

function encode(encoder::Thermometer, X::AbstractMatrix{T}) where {T <: Real}
    pattern = Array{Bool,3}(undef, encoder.resolution, size(X, 2), size(X, 1))

    encode!(encoder, X, pattern)
end

end