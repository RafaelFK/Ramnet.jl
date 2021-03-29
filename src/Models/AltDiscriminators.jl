module AltDiscriminators

using ..AltNodes
using ..Partitioners: partition, indices_to_segment_offset
using ..Encoders

using ..Loss
using ..Optimizers

import ..AbstractModel, ..train!, ..predict, ..reset!

using StaticArrays

using Base.Iterators:zip

abstract type AbstractDiscriminator{D,N <: AbstractNode{D},T,E <: AbstractEncoder{T}} <: AbstractModel end

struct RegressionDiscriminator{D,T,E} <: AbstractDiscriminator{D,RegressionNode{D},T,E}
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

function RegressionDiscriminator{D}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,Int}=nothing, γ::Float64=1.0) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [RegressionNode{D}(;γ) for _ in 1:cld(input_len * res, n)]

    RegressionDiscriminator{D,T,E}(input_len, n, encoder, partitioner, segments, offsets, nodes, zeros(Float64, D), zeros(Float64, D))
end

# TODO: Add Epochs
struct FunctionalDiscriminator{D,T,E <: AbstractEncoder{T}} <: AbstractDiscriminator{D,FunctionalNode{D},T,E}
    input_len::Int
    n::Int
    encoder::E
    partitioner::Symbol
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{FunctionalNode{D}}
    cum_weight::Vector{Float64}
end

function FunctionalDiscriminator{D}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,Int}=nothing) where {D,T,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [FunctionalNode{D}() for _ in 1:cld(input_len * res, n)]

    FunctionalDiscriminator{D,T,E}(input_len, n, encoder, partitioner, segments, offsets, nodes, zeros(Float64, D))
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

function reset!(d::M; seed::Union{Nothing,Int}=nothing) where {M <: AbstractDiscriminator}
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

# function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::StaticArray{Tuple{D},Float64,1}) where {D,T}
#     length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))

#     y_pred = predict(d, x) |> SizedVector{D}
#     for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
#         train!(node, encode(d.encoder, x, segments, offsets), -d.η * grad(d.loss, y, y_pred))
#     end

#     return nothing
# end

function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, grad::StaticArray{Tuple{D},Float64,1}) where {D,T}
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(
            node,
            encode(d.encoder, x, segments, offsets),
            -grad
        )
    end
end

function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, opt::FunctionalOptimizer{L}, x::AbstractVector{T}, y::StaticArray{Tuple{D},Float64,1}) where {D,T,L}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))

    y_pred = predict(d, x)
    gradient = -learning_rate(opt) * Optimizers.grad(opt, y, y_pred)

    train!(d, x, gradient)

    return nothing
end

# TODO: Deprecate this train methods that take as input an optimizer. Make a optimize function in the Optimizers module
# The following three functions could be generalized if the regression discriminator would also take an optimizer
function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, opt::FunctionalOptimizer{L}, x::AbstractVector{T}, y::Float64) where {D,T,L}
    train!(d, opt, x, SA_F64[y])
end

function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, opt::FunctionalOptimizer{L}, X::AbstractMatrix{T}, Y::AbstractMatrix{Float64}) where {D,T,L}
    for _ in 1:max_epochs(opt)
        for (x, y) in zip(eachcol(X), eachcol(Y))
            train!(d, opt, x, y)
        end
    end
end

function train!(d::FunctionalDiscriminator{D,T,<:AbstractEncoder{T}}, opt::FunctionalOptimizer{L}, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {D,T,L}
    for _ in 1:max_epochs(opt)
        for (x, target) in zip(eachcol(X), y)
            train!(d, opt, x, target)
        end
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


end

