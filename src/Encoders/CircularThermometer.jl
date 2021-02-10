"""
    CircularThermometer{T <: Real} <: AbstractEncoder

Circular thermometer encoder, used to represent real values as binary patterns.
"""
struct CircularThermometer{T <: Real} <: AbstractEncoder{T}
    min::Vector{T}
    max::Vector{T}
    # Precomputed constants
    A::Vector{Float64} # resolution / (max - min)
    B::Vector{Float64} # min * A
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

        A = resolution / (max - min)
        B = min * A

        block_len = fld(resolution, 2)

        new([min], [max], [A], [B], resolution, block_len)
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

        A = resolution ./ (max - min)
        B = min .* A

        block_len = fld(resolution, 2)

        new(min, max, A, B, resolution, block_len)
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

# =============================== Experimental =============================== #

function encode(encoder::CircularThermometer{T}, x::AbstractVector{T}, segment::Int, offset::Int) where {T <: Real}
    res = encoder.resolution
    block_len = encoder.block_len
    
    if length(encoder.min) != 1
        A = encoder.A[segment]
        B = encoder.B[segment]
    else
        A = encoder.A[1]
        B = encoder.B[1]
    end

    value = x[segment]

    shifts = floor(Int, value * A - B)
    if shifts < 0
        shifts = 0
    elseif shifts > res
        shifts = 5
    end

    shifted_offset = offset - shifts + 1

    if shifted_offset == 0
        return false
    elseif shifted_offset > 0
        return shifted_offset ≤ block_len
    else
        return (res + shifted_offset) ≤ block_len
    end
end

# TODO: This is the same as the simple thermometer. Generalize
function encode(encoder::CircularThermometer{T}, x::AbstractVector{T}, segments::AbstractVector{Int}, offsets::AbstractVector{Int}) where {T <: Real}
    value = zero(UInt)

    for (i, (segment, offset)) in enumerate(zip(segments, offsets))
        value += encode(encoder, x, segment, offset) << (i - 1)
    end

    return value
end

function pattern(encoder::CircularThermometer{T}, x::AbstractVector{T}) where {T <: Real}
    res = resolution(encoder)
    out = Vector{Bool}(undef, length(x) * res)

    for i in eachindex(x)
        for j in 0:res - 1
            out[(i - 1) * res + j + 1] = encode(encoder, x, i, j)
        end
    end

    return out
end

export pattern
