using Random
using Base.Iterators:enumerate

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

# =============================== Experimental =============================== #
function random_tuples(len::Int, n::Int; seed=Union{Nothing,Int} = nothing)
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    len < 0 && throw(DomainError(width, "Length must be non-negative"))
    n < 1 && throw(DomainError(n, "Partition may not be less then one"))
    
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 

    ordering = randperm(rng, len)

    return collect(Iterators.partition(ordering, n))
end

# TODO: indices length must match input_len times res
# function indices_to_segment_offset(indices::AbstractVector{Int}, input_len::Int, res::Int)
#     len = input_len * res
    
#     segments = Vector{Int}(undef, len)
#     offsets  = Vector{Int}(undef, len)

#     for (e, i) in enumerate(indices)
#         s, offset = fldmod(i - 1, res)

#         segments[e] = s + 1
#         offsets[e] = offset
#     end

#     return segments, offsets
# end

function random_tuples_segment_offset(input_len::Int, res::Int; seed::Union{Nothing,Int}=nothing)
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    input_len ≤ 0 && throw(DomainError(input_len, "Input length must be greater then zero"))

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 
    
    len = input_len * res
    ordering = randperm(rng, len)
    segments = Vector{Int}(undef, len)
    offsets  = Vector{Int}(undef, len)

    for i in 1:len
        s, offset = fldmod(i - 1, res)
        segment = s + 1

        segments[i] = segment
        offsets[i] = offset
    end

    return segments[ordering], offsets[ordering]
end

function uniform_random_tuples(input_len::Int, res::Int, ::Int; seed::Union{Nothing,Int}=nothing)
    !isnothing(seed) && seed < 0 && throw(DomainError(seed, "Seed must be non-negative"))
    input_len ≤ 0 && throw(DomainError(input_len, "Input length must be greater then zero"))

    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed) 
    
    len = input_len * res
    
    return randperm(rng, len)
end

function significance_aware_random_tuples(input_len::Int, res::Int, n::Int; seed::Union{Nothing,Int}=nothing)
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