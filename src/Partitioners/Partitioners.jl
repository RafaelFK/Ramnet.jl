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
    significance_aware_random_tuples,
    indices_to_segment_offset

include("LinearPartitioner.jl")
export LinearPartitioner

end