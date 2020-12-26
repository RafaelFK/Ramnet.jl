struct Thermometer{T <: Real} <: AbstractEncoder
    min::T
    max::T
    resolution::Int # TODO: This could be unsigned

    function Thermometer{T}(min, max, resolution::Int) where {T <: Real}
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

Thermometer(min::T, max::T, resolution::Int) where {T <: Real} = Thermometer{T}(min, max, resolution)
Thermometer(min::Real, max::Real, resolution::Int) = Thermometer(promote(min, max)..., resolution)
Thermometer(::Type{T}, min::Real, max::Real, resolution) where {T <: Real} = Thermometer(convert.(T, (min, max))..., resolution)

function encode!(encoder::Thermometer{T}, x::T, pattern::AbstractVector{Bool}) where {T <: Real}
    validate_input_output(encoder, x, pattern)
  
    for i in eachindex(pattern)
        pattern[i] = x > encoder.min + (i - 1) * (encoder.max - encoder.min) / encoder.resolution
    end

    pattern
end
