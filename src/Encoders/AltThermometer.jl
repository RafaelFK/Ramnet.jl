using Base.Iterators: enumerate, zip

# TODO: I could get rid of a lot of if-else checks if min, max and rel_extrema_diff
#       always had the same shape as the input. I order to do that, however, the
#       encoder would need to now the input shape, which would break a lot of things
"""
    AltThermometer{T <: Real} <: AbstractEncoder

AltThermometer encoder, used to represent real values as binary patterns.
"""
struct AltThermometer{Size,T <: Real} <: AbstractEncoder{T}
    min::Vector{T}
    max::Vector{T}
    rel_extrema_diff::Vector{Float64} # (max - min) / resolution
    resolution::Int # TODO: This could be unsigned

    """
        AltThermometer{T}(min::T, max::T, resolution::Int) where {T <: Real}

    Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
    numerical limits of encoded values and `resolution` specifies the number of
    bits dedicated to the encoding of a scalar. Values to be encoded must be of
    type T.
    """
    function AltThermometer{T}(min::T, max::T, resolution::Int) where {T <: Real}
        max ≤ min && throw(ArgumentError(
            "the maximum of the AltThermometer may not be lesser or equal to its minimum"
        ))

        resolution < 1 && throw(DomainError(
            resolution,
            "the AltThermometer's resolution must be greater then zero"
        ))

        new([min], [max], [(max - min) / resolution], resolution)
    end

    """
        AltThermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}

    Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
    numerical limits of encoded values and `resolution` specifies the number of
    bits dedicated to the encoding of a scalar. Values to be encoded are
    expected to have the same shape as `min` and `max`. Values to be encoded
    must be of type T.
    """
    function AltThermometer{T}(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}
        length(min) != length(max) && throw(DimensionMismatch(
            "min and max vectors must have the same length"
        ))

        !all(min .≤ max) && throw(ArgumentError(
            "the maximum of the AltThermometer may not be lesser or equal to its minimum"
        ))

        resolution < 1 && throw(DomainError(
            resolution,
            "the AltThermometer's resolution must be greater then zero"
        ))

        new(min, max, (max - min) / resolution, resolution)
    end
end

"""
    AltThermometer(min::T, max::T, resolution::Int) where {T <: Real}

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be of type
T.
"""
AltThermometer(min::T, max::T, resolution::Int) where {T <: Real} = AltThermometer{T}(min, max, resolution)

"""
    AltThermometer(min::Real, max::Real, resolution::Int)

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. Values to be encoded must be same as
the promotion of `min` and `max` types.
"""
AltThermometer(min::Real, max::Real, resolution::Int) = AltThermometer(promote(min, max)..., resolution)

"""
    AltThermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real}

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of
bits dedicated to the encoding of a scalar. First argument determines the type
of values to be encoded.
"""
AltThermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real} = AltThermometer(convert.(T, (min, max))..., resolution)


"""
    AltThermometer(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real}

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. Values to be encoded must be of type T.
"""
AltThermometer(min::AbstractVector{T}, max::AbstractVector{T}, resolution::Int) where {T <: Real} = AltThermometer{T}(min, max, resolution)

"""
    AltThermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int)

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. Values to be encoded must be same as the
promotion of `min` and `max` types.
"""
AltThermometer(min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution::Int) = AltThermometer(promote(min, max)..., resolution)

"""
    AltThermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution)

Constructs a AltThermometer encoder. `min` and `max` denotes the admissible
numerical limits of encoded values and `resolution` specifies the number of bits
dedicated to the encoding of a scalar. Values to be encoded are expected to have
the same shape as `min` and `max`. First argument determines the type of values
to be encoded.
"""
AltThermometer(::Type{T}, min::AbstractVector{<:Real}, max::AbstractVector{<:Real}, resolution) where {T <: Real} = AltThermometer(convert.(Vector{T}, (min, max))..., resolution)

function encode!(encoder::AltThermometer{T}, min::T, max::T, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    for i in eachindex(pattern)
        pattern[i] = x > min + (i - 1) * (max - min) / encoder.resolution
    end

    pattern
end

function encode!(encoder::AltThermometer{T}, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    validate_input_output(encoder, x, pattern)

    encode!(encoder, encoder.min[1], encoder.max[1], x, pattern)
end

# =============================== Experimental =============================== #

# TODO: This if-else will be check an absurd amount of times when called by the function bellow.
#       Maybe make specialized versions passing in the appropriate min  and extrema_diff values
function encode(encoder::AltThermometer{T}, x::AbstractVector{T}, segment::Int, offset::Int) where {T <: Real}
    if length(encoder.min) != 1
        min = encoder.min[segment]
        rel_extrema_diff = encoder.rel_extrema_diff[segment]
    else
        min = encoder.min[1]
        rel_extrema_diff = encoder.rel_extrema_diff[1]
    end

    return x[segment] > min + offset * rel_extrema_diff
end

function encode(encoder::AltThermometer{T}, x::AbstractVector{T}, segments::AbstractVector{Int}, offsets::AbstractVector{Int}) where {T <: Real}
    value = zero(UInt)

    for (i, (segment, offset)) in enumerate(zip(segments, offsets))
        value += encode(encoder, x, segment, offset) << (i - 1)
    end

    return value
end

# TODO: This might no longer be useful
function encode(encoder::AltThermometer{T}, x::AbstractVector{T}, index::Int) where {T <: Real}
    d, offset = fldmod(index - 1, encoder.resolution)
    segment = d + 1
    
    return encode(encoder, x, segment, offset)
end
