# TODO: Consider other concept for the names here instead of mapping (ex. partitioning)
module Partitioners

export AbstractPartitioner, partition

abstract type AbstractPartitioner end

struct PartitionerIterator{P <: AbstractPartitioner,T <: AbstractVecOrMat}
    partitioner::P
    X::T
end

# In these generic functions I'm implicitly enforcing that all
# AbstractPartitioner subtypes have `width` and `n` fields. Maybe it would be
# better to enforce in a more explicit manner through traits

function partition(partitioner::P, X::T) where {P <: AbstractPartitioner,T <: AbstractVector}
    length(X) != partitioner.width && throw(
      DimensionMismatch("Expected length of X to be $(partitioner.width), got $(length(X))"))

    return PartitionerIterator{P,T}(partitioner, X)
end

function partition(partitioner::P, X::T) where {P <: AbstractPartitioner,T <: AbstractMatrix}
    size(X, 2) != partitioner.width && throw(
    DimensionMismatch("Expected the number of columns of X to be $(partitioner.width), got $(size(X, 2))"))

    return PartitionerIterator{P,T}(partitioner, X)
end

Base.length(partitioner::P) where {P <: AbstractPartitioner} = ceil(Int, partitioner.width / partitioner.n)

function Base.length(itr::PartitionerIterator{P,T}) where {P <: AbstractPartitioner,T <: AbstractVecOrMat}
    # ceil(Int, size(itr.X)[end] / itr.partitioner.n)
    length(itr.partitioner)
end

Base.eltype(::Type{PartitionerIterator{P,T}}) where {P <: AbstractPartitioner,T <: AbstractVecOrMat} = T

include("RandomPartitioner.jl")
export RandomPartitioner,
    random_partitioning,
    random_tuples,
    random_tuples_segment_offset,
    uniform_random_tuples,
    significance_aware_random_tuples
    # indices_to_segment_offset

include("LinearPartitioner.jl")
export LinearPartitioner

# ============================================================================ #
# ------------------------- Simplifying partitioning ------------------------- #

partition(scheme::Symbol, args...; kargs...) = partition(Val(scheme), args...; kargs...)

function partition(::Val{S}, args...; kargs...) where {S}
    throw(DomainError(S, "unknown partitioning scheme"))
end

function partition(::Val{:uniform_random}, input_len::Int, res::Int, ::Int; seed::Union{Nothing,Int}=nothing)
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    input_len ≤ 0 && throw(DomainError(input_len, "Input length must be greater then zero"))

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 
    
    len = input_len * res
    
    return randperm(rng, len)
end

function partition(::Val{:significance_aware_random}, input_len::Int, res::Int, n::Int; seed::Union{Nothing,Int}=nothing)
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    input_len ≤ 0 && throw(DomainError(input_len, "Input length must be greater then zero"))

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 
    
    data = first.(sort(
            shuffle(
                rng,
                collect(enumerate(repeat(1:res, input_len)))
            );
            by=last
        ))

    len = input_len * res

    indices = Vector{Int}(undef, len)
    out = 1
    for i in 1:len
            indices[out] = data[i]

        out += res
    if out > len
            out = (out % len) + 1
        end
    end

    return indices
end

function indices_to_segment_offset(indices::AbstractVector{Int}, input_len::Int, res::Int)
    len = input_len * res
    
    segments = Vector{Int}(undef, len)
    offsets  = Vector{Int}(undef, len)

    for (e, i) in enumerate(indices)
        s, offset = fldmod(i - 1, res)

        segments[e] = s + 1
        offsets[e] = offset
    end

    return segments, offsets
end

export indices_to_segment_offset

end