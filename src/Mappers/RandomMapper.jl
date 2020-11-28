# TODO: Consider other concept for the names here instead of mapping (ex. partitioning)

using Random

struct RandomMapper <: Mapper
    width::Int
    n::Int
    masks::Vector{Vector{Int}}

    function RandomMapper(width::Int, n::Int; seed::Union{Nothing,Int}=nothing)
        !isnothing(seed) && seed < 0 && throw(ArgumentError("Seed must be non-negative"))
        width < 0 && throw(ArgumentError("Width must be non-negative"))
        n < 1 && throw(ArgumentError("Partition may not be less then one"))
      
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 

        ordering = randperm(rng, width)
        nmasks = ceil(Int, width / n)

        # Could I avoid allocating the masks?
        masks = collect(Iterators.partition(ordering, n))

        new(width, n, masks)
    end
end

struct RandomMapperIterator{T <: AbstractVecOrMat}
    mapper::RandomMapper
    X::T
end

function Base.map(mapper::RandomMapper, X::T) where {T <: AbstractVector}
    length(X) != mapper.width && throw(
        DimensionMismatch("Expected length of X to be $(mapper.width), got $(length(X))"))

    return RandomMapperIterator{T}(mapper, X)
end

function Base.map(mapper::RandomMapper, X::T) where {T <: AbstractMatrix}
    size(X, 2) != mapper.width && throw(
      DimensionMismatch("Expected the number of columns of X to be $(mapper.width), got $(size(X, 2))"))

    return RandomMapperIterator{T}(mapper, X)
end

Base.length(mapper::RandomMapper) = length(mapper.masks)

Base.length(itr::RandomMapperIterator) = length(itr.mapper.masks)

Base.eltype(::Type{RandomMapperIterator{T}}) where T <: AbstractVecOrMat = T

function Base.iterate(itr::RandomMapperIterator{<:AbstractVector}, state=1)
    state > length(itr.mapper.masks) && return nothing

    return itr.X[itr.mapper.masks[state]], state + 1
end

function Base.iterate(itr::RandomMapperIterator{<:AbstractMatrix}, state=1)
    state > length(itr.mapper.masks) && return nothing

    return itr.X[:, itr.mapper.masks[state]], state + 1
end

function random_mapping(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractVector}
    return Base.map(RandomMapper(length(X), n; seed), X)
end

function random_mapping(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractMatrix}
    return Base.map(RandomMapper(size(X, 2), n; seed), X)
end