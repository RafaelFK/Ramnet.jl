"""
    Thermometer{T <: Real} <: AbstractEncoder

Thermometer encoder, used to represent real values as binary patterns.
"""
struct Thermometer{T <: Real} <: AbstractEncoder
    min::Union{T,AbstractVector{T}}
    max::Union{T,AbstractVector{T}}
    resolution::Int # TODO: This could be unsigned

    """
        Thermometer{T}(min::T, max::T, resolution::Int) where {T <: Real}

    Constructs a thermometer encoder. `min` and `max` denotes the admissible
    numerical limits of encoded values and `resolution` specifies the number of
    bits dedicated to the encoding of a scalar. Values to be encoded must be of
    type T.
    """
    function Thermometer{T}(min::T, max::T, resolution::Int) where {T <: Real}
        max ≤ min && throw(ArgumentError(
            "the maximum of the thermometer may not be lesser or equal to its minimum"
        ))

        resolution < 1 && throw(DomainError(
            resolution,
            "the thermometer's resolution must be greater then zero"
        ))

        new(min, max, resolution)
    end

    """
        Thermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}

    Constructs a thermometer encoder. `min` and `max` denotes the admissible
    numerical limits of encoded values and `resolution` specifies the number of
    bits dedicated to the encoding of a scalar. Values to be encoded are
    expected to have the same shape as `min` and `max`. Values to be encoded
    must be of type T.
    """
    function Thermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}
        length(min) != length(max) && throw(DimensionMismatch(
            "min and max vectors must have the same length"
        ))

        !all(min .≤ max) && throw(ArgumentError(
            "the maximum of the thermometer may not be lesser or equal to its minimum"
        ))

        resolution < 1 && throw(DomainError(
            resolution,
            "the thermometer's resolution must be greater then zero"
        ))

        new(min, max, resolution)
    end
end

"""
    Thermometer(min::T, max::T, resolution::Int) where {T <: Real}

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be of type
T.
"""
Thermometer(min::T, max::T, resolution::Int) where {T <: Real} = Thermometer{T}(min, max, resolution)

"""
    Thermometer(min::Real, max::Real, resolution::Int)

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be same as
the promotion of `min` and `max` types.
"""
Thermometer(min::Real, max::Real, resolution::Int) = Thermometer(promote(min, max)..., resolution)

"""
    Thermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real}

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. First argument determines the type
of values to be encoded.
"""
Thermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real} = Thermometer(convert.(T, (min, max))..., resolution)


"""
    Thermometer(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. Values to be encoded must be of type T.
"""
Thermometer(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real} = Thermometer{T}(min, max, resolution)

"""
    Thermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int)

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. Values to be encoded must be same as the
promotion of `min` and `max` types.
"""
Thermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int) = Thermometer(promote(min, max)..., resolution)

"""
    Thermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution)

Constructs a thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. First argument determines the type of values
to be encoded.
"""
Thermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution) where {T <: Real} = Thermometer(convert.(Vector{T}, (min, max))..., resolution)

function encode!(encoder::Thermometer{T}, min::T, max::T, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    for i in eachindex(pattern)
        pattern[i] = x > min + (i - 1) * (max - min) / encoder.resolution
    end

    pattern
end

function encode!(encoder::Thermometer{T}, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    validate_input_output(encoder, x, pattern)

    encode!(encoder, encoder.min[1], encoder.max[1], x, pattern)
end
