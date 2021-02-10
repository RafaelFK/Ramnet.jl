using Base.Iterators: enumerate

struct IdentityEncoder <: AbstractEncoder{Bool} end

function encode(::IdentityEncoder, x::AbstractVector{Bool}, index::Int)
    return x[index]
end

function encode(::IdentityEncoder, x::AbstractVector{Bool}, indices::AbstractVector{Int})
    value = zero(UInt)

    for (i, index) in enumerate(indices)
        value += x[index] << (i - 1)
    end

    return value
end