"""
    CircularThermometer{T <: Real} <: AbstractEncoder

Circular thermometer encoder, used to represent real values as binary patterns.
"""
struct CircularThermometer{T <: Real} <: AbstractEncoder
    min::Union{T,AbstractVector{T}}
    max::Union{T,AbstractVector{T}}
    resolution::Int # TODO: This could be unsigned
    block_len::Int

    """
        CircularThermometer{T}(min, max, resolution::Int) where {T <: Real}

    Constructs a circular thermometer encoder. `min` and `max` denotes the
    admissible numerical limits of encoded values and `resolution` specifies the
    number of bits dedicated to the encoding of a scalar. Values to be encoded
    must be of type T.
    """
    function CircularThermometer{T}(min, max, resolution::Int) where {T <: Real}
        max <= min && throw(ArgumentError(
            "the maximum of the circular thermometer may not be lesser or equal to its minimum"
        ))

        resolution <= 1 && throw(DomainError(
            resolution,
            "circular thermometer's resolution must be greater then one"
        ))

        new(min, max, resolution, floor(Int, resolution / 2))
    end

    """
    CircularThermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}

    Constructs a circular thermometer encoder. `min` and `max` denotes the
    admissible numerical limits of encoded values and `resolution` specifies the
    number of bits dedicated to the encoding of a scalar. Values to be encoded
    are expected to have the same shape as `min` and `max`. Values to be encoded
    must be of type T.
    """
    function CircularThermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}
        length(min) != length(max) && throw(DimensionMismatch(
            "min and max vectors must have the same length"
        ))

        !all(min .≤ max) && throw(ArgumentError(
            "the maximum of the circular thermometer may not be lesser or equal to its minimum"
        ))

        resolution <= 1 && throw(DomainError(
            resolution,
            "circular thermometer's resolution must be greater then one"
        ))

        new(min, max, resolution, floor(Int, resolution / 2))
    end
end

"""
    CircularThermometer(min::T, max::T, resolution::Int) where {T <: Real}

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be of type
T.
"""
CircularThermometer(min::T, max::T, resolution::Int) where {T <: Real} = CircularThermometer{T}(min, max, resolution)

"""
    CircularThermometer(min::Real, max::Real, resolution::Int)

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be same as
the promotion of `min` and `max` types.
"""
CircularThermometer(min::Real, max::Real, resolution::Int) = CircularThermometer(promote(min, max)..., resolution)

"""
    CircularThermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real}

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. First argument determines the type
of values to be encoded.
"""
CircularThermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real} = CircularThermometer(convert.(T, (min, max))..., resolution)


"""
    CircularThermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real}

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. First argument determines the type
of values to be encoded.
"""
CircularThermometer(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real} = CircularThermometer{T}(min, max, resolution)

"""
    CircularThermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int)

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. Values to be encoded must be same as the
promotion of `min` and `max` types.
"""
CircularThermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int) = CircularThermometer(promote(min, max)..., resolution)

"""
    CircularThermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution)

Constructs a circular thermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. First argument determines the type of values
to be encoded.
"""
CircularThermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution) where {T <: Real} = CircularThermometer(convert.(Vector{T}, (min, max))..., resolution)

function encode!(encoder::CircularThermometer{T}, min::T, max::T, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    if x < min
        x = min
    elseif x > max
        x = max
    end

    shifts = floor(Int, (x - min) / (max - min) * encoder.resolution)
    l = shifts % encoder.resolution
    u = (shifts + encoder.block_len + 1) % encoder.resolution
    u = (u == 0) ? encoder.resolution : u
    
    for i in eachindex(pattern)
        if l < u
            pattern[i] = (l < i < u) ? 1 : 0
        else
            pattern[i] = (1 ≤ i < u || l < i ≤ encoder.resolution) ? 1 : 0
        end
    end

    pattern
end

function encode!(encoder::CircularThermometer{T}, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    validate_input_output(encoder, x, pattern)

    encode!(encoder, encoder.min[1], encoder.max[1], x, pattern)
end
