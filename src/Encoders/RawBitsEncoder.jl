using Base.Iterators: enumerate

struct RawBitsEncoder{T} <: AbstractEncoder{T} end

resolution(::RawBitsEncoder{T}) where {T <: Real} = 8 * sizeof(T)

# TODO: Validate offset. Must be no larger then 8*sizeof(T)
function encode(::RawBitsEncoder{Float64}, x::AbstractVector{Float64}, segment::Int, offset::Int)
    return (reinterpret(UInt64, x[segment]) & (1 << offset)) > 0
end

function encode(::RawBitsEncoder{UInt16}, x::AbstractVector{UInt16}, segment::Int, offset::Int)
    return (x[segment] & (1 << offset)) > 0
end

# TODO: This is kinda the same for all encoders. Could generalize 
function encode(encoder::RawBitsEncoder{T}, x::AbstractVector{T}, segments::AbstractVector{Int}, offsets::AbstractVector{Int}) where {T <: Real}
    value = zero(UInt)

    for (i, (segment, offset)) in enumerate(zip(segments, offsets))
        value += encode(encoder, x, segment, offset) << (i - 1)
    end

    return value
end