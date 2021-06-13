module AltDiscriminators

using ..AltNodes
using ..Partitioners: partition, indices_to_segment_offset
using ..Encoders

# using ..Loss
# using ..Optimizers

import ..AbstractModel, ..train!, ..predict, ..reset!

using StaticArrays

using Base.Iterators:zip

abstract type AbstractDiscriminator{D,N <: AbstractNode{D},T,E <: AbstractEncoder{T}} <: AbstractModel end

mutable struct RegressionDiscriminator{D,T,E} <: AbstractDiscriminator{D,RegressionNode{D},T,E}
    input_len::Int
    n::Int
    encoder::E
    partitioner::Symbol
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{RegressionNode{D}}
    running_numerator::Vector{Float64}
    running_denominator::Vector{Float64}
end

function RegressionDiscriminator{D}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,UInt}=nothing, γ::Float64=1.0) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [RegressionNode{D}(;γ) for _ in 1:cld(input_len * res, n)]

    RegressionDiscriminator{D,T,E}(input_len, n, encoder, partitioner, segments, offsets, nodes, zeros(Float64, D), zeros(Float64, D))
end

# TODO: Add Epochs
mutable struct FunctionalDiscriminator{D,T,E <: AbstractEncoder{T}} <: AbstractDiscriminator{D,FunctionalNode{D},T,E}
    input_len::Int
    n::Int
    encoder::E
    partitioner::Symbol
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{FunctionalNode{D}}
    cum_weight::Vector{Float64}
end

function FunctionalDiscriminator{D}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,UInt}=nothing) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [FunctionalNode{D}() for _ in 1:cld(input_len * res, n)]

    FunctionalDiscriminator{D,T,E}(input_len, n, encoder, partitioner, segments, offsets, nodes, zeros(Float64, D))
end

mutable struct RegularizedFunctionalDiscriminator{D,T,E <: AbstractEncoder{T}} <: AbstractDiscriminator{D,RegularizedFunctionalNode{D},T,E}
    input_len::Int
    n::Int
    sample_count::UInt
    encoder::E
    partitioner::Symbol
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{RegularizedFunctionalNode{D}}
    cum_weight::Vector{Float64}
end

function RegularizedFunctionalDiscriminator{D}(input_len::Int, n::Int, α::Float64, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,UInt}=nothing) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [RegularizedFunctionalNode{D}(; α) for _ in 1:cld(input_len * res, n)]

    RegularizedFunctionalDiscriminator{D,T,E}(input_len, n, zero(UInt), encoder, partitioner, segments, offsets, nodes, zeros(Float64, D))
end


mutable struct AdaptiveFunctionalDiscriminator{D,T,E <: AbstractEncoder{T}} <: AbstractDiscriminator{D,AdaptiveFunctionalNode{D},T,E}
    input_len::Int
    n::Int
    sample_count::UInt
    encoder::E
    partitioner::Symbol
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{AdaptiveFunctionalNode{D}}
    cum_weight::Vector{Float64}
    cum_v::Vector{Float64}
end

function AdaptiveFunctionalDiscriminator{D}(input_len::Int, n::Int, λ::Float64, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,UInt}=nothing) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [AdaptiveFunctionalNode{D}(; λ) for _ in 1:cld(input_len * res, n)]

    AdaptiveFunctionalDiscriminator{D,T,E}(input_len, n, zero(UInt), encoder, partitioner, segments, offsets, nodes, zeros(Float64, D), zeros(Float64, D))
end

# ============================= Generic functions ============================ #

function train!(d::M, x::AbstractVector{T}, y::Float64) where {T,M <: AbstractDiscriminator{1,<:AbstractNode{1},T,<:AbstractEncoder{T}}}
    train!(d, x, SA_F64[y])
end

function train!(d::M, x::AbstractVector{T}, y::AbstractVector{Float64}) where {D,T,M <: AbstractDiscriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T}}}
    train!(d, x, SizedVector{D}(y))
end

function train!(d::M, X::AbstractMatrix{T}, Y::AbstractMatrix{Float64}) where {D,T,M <: AbstractDiscriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T}}}
    for (x, y) in zip(eachcol(X), eachcol(Y))
        train!(d, x, y)
    end
end

function train!(d::M, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {D,T,M <: AbstractDiscriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T}}}
    for (x, target) in zip(eachcol(X), y)
        train!(d, x, target)
    end
end

function predict(d::M, X::AbstractMatrix{T}) where {D,T,M <: AbstractDiscriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T}}}
    out = Matrix{Float64}(undef, D, size(X, 2))

    for (i, x) in zip(axes(out, 2), eachcol(X))
        out[:, i] = predict(d, x)
    end

    out
end

function reset!(d::M; seed::Union{Nothing,UInt}=nothing) where {M <: AbstractDiscriminator}
    if !isnothing(seed)
        res = resolution(d.encoder)
        indices = partition(d.partitioner, d.input_len, res, d.n; seed)
        d.segments, d.offsets = indices_to_segment_offset(indices, d.input_len, res)
    end

    for node in d.nodes
        reset!(node)
    end

    nothing
end

# =========================== Specialized functions ========================== #
# ------------------------- Regression Discriminator ------------------------- #

function Base.show(io::IO, d::RegressionDiscriminator{D,T,E}) where {D,T,E}
    print(
        io,
        """Regression Discriminator
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ γ: $(d.nodes[1].γ)
        ├ Encoder type: $(E)
        └ Partitioning scheme: $(d.partitioner)"""
    )
end

function train!(d::RegressionDiscriminator{D,T,E}, x::AbstractVector{T}, y::StaticArray{Tuple{D},Float64,1}) where {D,T,E}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y)
    end

    return nothing
end

# TODO: Have γ be a parameter and make a specialized version for the case γ = 1.0
function predict(d::RegressionDiscriminator{D,T,E}, x::AbstractVector{T}) where {D,T,E}
    fill!(d.running_numerator, 0.0)
    fill!(d.running_denominator, 0.0)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        el = predict(node, key)

        d.running_numerator .+= el.value

        if node.γ != 1.0
            @. d.running_denominator += (1 - node.γ^el.count) / (1 - node.γ)
        else
            @. d.running_denominator += el.count
        end
    end

    return map(d.running_numerator, d.running_denominator) do num, den
        den == 0.0 ? 0.0 : num / den
    end
end

# ------------------------- Functional Discriminator ------------------------- #
function Base.show(io::IO, d::FunctionalDiscriminator{D,T,E}) where {D,T,E}
    print(
        io,
        """Functional Discriminator{$D}
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ Encoder type: $(E)
        └ Partitioning scheme: $(d.partitioner)"""
    )
end

function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, grad::StaticArray{Tuple{D},Float64,1}) where {D,T}
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(
            node,
            encode(d.encoder, x, segments, offsets),
            -grad
        )
    end
end

function predict(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {D,T}
    fill!(d.cum_weight, 0.0)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        
        # I could avoid creating temp arrays here 
        weight = predict(node, key)

        d.cum_weight .+= weight
    end

    return d.cum_weight ./ length(d.nodes)
end

# ------------------- Regularized Functional Discriminator ------------------- #
function Base.show(io::IO, d::RegularizedFunctionalDiscriminator{D,T,E}) where {D,T,E}
    print(
        io,
        """Regularized Functional Discriminator{$D}
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ α: $(first(d.nodes).α)
        ├ Encoder type: $(E)
        └ Partitioning scheme: $(d.partitioner)"""
    )
end

function train!(d::RegularizedFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, grad::StaticArray{Tuple{D},Float64,1}) where {D,T}
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(
            node,
            encode(d.encoder, x, segments, offsets),
            -grad,
            d.sample_count
        )
    end

    d.sample_count += 1

    nothing
end

# A lot of overlap with the previous function
function train!(d::RegularizedFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, grads::AbstractMatrix{Float64}) where {D,T}
    for (x, grad) in zip(eachcol(X), eachcol(grads))
        for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
            train!(
                node,
                encode(d.encoder, x, segments, offsets),
                -grad,
                d.sample_count
            )
        end
    end

    d.sample_count += 1

    nothing
end

function predict(d::RegularizedFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {D,T}
    fill!(d.cum_weight, 0.0)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        
        # I could avoid creating temp arrays here 
        weight = predict(node, key, d.sample_count).value

        d.cum_weight .+= weight
    end

    return d.cum_weight ./ length(d.nodes)
end

# --------------------- Adaptive Functional Discriminator -------------------- #
function Base.show(io::IO, d::AdaptiveFunctionalDiscriminator{D,T,E}) where {D,T,E}
    print(
        io,
        """Adaptive Functional Discriminator{$D}
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ λ: $(first(d.nodes).λ)
        ├ Encoder type: $(E)
        └ Partitioning scheme: $(d.partitioner)"""
    )
end

function train!(d::AdaptiveFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, grad::StaticArray{Tuple{D},Float64,1}, trace_update::StaticArray{Tuple{D},Float64,1}) where {D,T}
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(
            node,
            encode(d.encoder, x, segments, offsets),
            -grad,
            -trace_update,
            d.sample_count
        )
    end

    d.sample_count += 1

    nothing
end

# A lot of overlap with the previous function
function train!(d::AdaptiveFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, grads::AbstractMatrix{Float64}) where {D,T}
    for (x, grad) in zip(eachcol(X), eachcol(grads))
        for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
            train!(
                node,
                encode(d.encoder, x, segments, offsets),
                -grad,
                -v,
                d.sample_count
            )
        end
    end

    d.sample_count += 1

    nothing
end

function predict(d::AdaptiveFunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {D,T}
    fill!(d.cum_weight, 0.0)
    fill!(d.cum_v, 0.0)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        
        # I could avoid creating temp arrays here 
        val = predict(node, key, d.sample_count).value

        d.cum_weight .+= val[1:D]
        d.cum_v .+= val[D + 1:end]
    end

    return (d.cum_weight ./ length(d.nodes), d.cum_v ./ length(d.nodes))
end

end

