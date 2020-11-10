module ramnet

using Random

export Discriminator, random_mapping, RandomMapper

struct RandomMapper{T <: VecOrMat}
    X::T
    masks::Vector{Vector{Int}}

    function RandomMapper(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: VecOrMat}
        !isnothing(seed) && seed < 0 && throw(ArgumentError("Seed must be non-negative"))
        n < 1 && throw(ArgumentError("Partition may not be less then one"))
        
        rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 
        len = size(X)[end]
        ordering = randperm(rng, len)
        nmasks = ceil(Int, len / n)

        # Could I avoid allocating the masks?
        masks = collect(Iterators.partition(ordering, n))

        new{T}(X, masks)
    end
end

Base.length(itr::RandomMapper) = length(itr.masks)

Base.eltype(::Type{RandomMapper{T}}) where T <: VecOrMat = T

function Base.iterate(itr::RandomMapper{<:AbstractVector}, state=1)
    state > length(itr.masks) && return nothing

    return itr.X[itr.masks[state]], state + 1
end

function Base.iterate(itr::RandomMapper{<:AbstractMatrix}, state=1)
    state > length(itr.masks) && return nothing

    return itr.X[:, itr.masks[state]], state + 1
end

function random_mapping(pattern::AbstractVector{Bool}, n::Int; seed::Union{Nothing,Int}=nothing)
    !isnothing(seed) && seed < 0 && throw(ArgumentError("Seed must be non-negative"))

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 

    ordering = randperm(rng, length(pattern))

    return Iterators.partition(pattern[ordering], n)
end

struct Discriminator
    address_size::Int
    nodes::Vector{Dict{BitVector,Int8}}

    function Discriminator(X::AbstractVector{<:AbstractVector{Bool}}, n::Int)
        n < 1 && throw(ArgumentError("Address size must not be less then 1"))

        # Create the list of nodes
        # In order to do that, I need to know the number of partitions
    end
end

# Discriminator() = Discriminator(Vector())

# function train!(disc::Discriminator, )

# end


end
