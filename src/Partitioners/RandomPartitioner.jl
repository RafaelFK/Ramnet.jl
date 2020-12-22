using Random

struct RandomPartitioner <: AbstractPartitioner
    width::Int
    n::Int
    masks::Vector{Vector{Int}}

    function RandomPartitioner(width::Int, n::Int; seed::Union{Nothing,Int}=nothing)
        !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
        width < 0 && throw(DomainError(width, "Width must be non-negative"))
        n < 1 && throw(DomainError(n, "Partition may not be less then one"))
      
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 

        ordering = randperm(rng, width)
        nmasks = ceil(Int, width / n)

        # Could I avoid allocating the masks?
        masks = collect(Iterators.partition(ordering, n))

        new(width, n, masks)
    end
end

function Base.iterate(itr::PartitionerIterator{RandomPartitioner,<:AbstractVector}, state=1)
    state > length(itr.partitioner.masks) && return nothing

    return itr.X[itr.partitioner.masks[state]], state + 1
end

function Base.iterate(itr::PartitionerIterator{RandomPartitioner,<:AbstractMatrix}, state=1)
    state > length(itr.partitioner.masks) && return nothing

    return itr.X[:, itr.partitioner.masks[state]], state + 1
end

function random_partitioning(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractVector}
    return partition(RandomPartitioner(length(X), n; seed), X)
end

function random_partitioning(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractMatrix}
    return partition(RandomPartitioner(size(X, 2), n; seed), X)
end