module AltDiscriminators

using ..AltNodes
using ..Partitioners: partition, indices_to_segment_offset
using ..Encoders

using ..Loss

import ..AbstractModel, ..train!, ..predict, ..reset!

using StaticArrays

using Base.Iterators:zip

# TODO: Add Epochs
struct Discriminator{D,N <: AbstractNode{D},T,E <: AbstractEncoder{T},L <: AbstractLoss} <: AbstractModel
    input_len::Int
    n::Int
    encoder::E
    partitioner::Symbol
    loss::L
    η::Float64
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{N}
end

function Discriminator{D,N}(input_len::Int, n::Int, encoder::E, loss::L, η::Float64, partitioner::Symbol=:uniform_random; seed::Union{Nothing,Int}=nothing, kargs...) where {D,N,T,E <: AbstractEncoder{T},L}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partition(partitioner, input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [N(;kargs...) for _ in 1:cld(input_len * res, n)]

    Discriminator{D,N,T,E,L}(input_len, n, encoder, partitioner, loss, η, segments, offsets, nodes)
end

## Aliases and convenience constructors
const RegressionDiscriminator{D} = Discriminator{D,RegressionNode{D}}
const FunctionalDiscriminator{D} = Discriminator{D,FunctionalNode{D}}

function Discriminator{D,RegressionNode{D}}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; seed::Union{Nothing,Int}=nothing, γ=1.0) where {D,T,E <: AbstractEncoder{T}}
    Discriminator{D,RegressionNode{D}}(input_len, n, encoder, NoneLoss(), 0.0, partitioner; seed, γ)
end

function Discriminator{D,FunctionalNode{D}}(input_len::Int, n::Int, encoder::E, partitioner::Symbol=:uniform_random; η, seed::Union{Nothing,Int}=nothing, loss::L=SquaredError()) where {D,T,E <: AbstractEncoder{T},L}
    Discriminator{D,FunctionalNode{D}}(input_len, n, encoder, loss, η, partitioner; seed)
end

# ============================= Generic functions ============================ #
function Base.show(io::IO, d::Discriminator{D,N,T,E,L}) where {D,N,T,E,L}
    print(
        io,
        """Discriminator
        ├ Node: $(N)
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ Encoder type: $(E)
        ├ Partitioning scheme: $(d.partitioner)
        └ Loss function: $(L)"""
    )
end

function train!(d::Discriminator{1,<:AbstractNode{1},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}, y::Float64) where {T,L}
    train!(d, x, SA_F64[y])
end

function train!(d::Discriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}, y::AbstractVector{Float64}) where {D,T,L}
    train!(d, x, SizedVector{D}(y))
end

function train!(d::Discriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T},L}, X::AbstractMatrix{T}, Y::AbstractMatrix{Float64}) where {D,T,L}
    for (x, y) in zip(eachcol(X), eachcol(Y))
        train!(d, x, y)
    end
end

function train!(d::Discriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T},L}, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {D,T,L}
    for (x, target) in zip(eachcol(X), y)
        train!(d, x, target)
    end
end

function predict(d::Discriminator{D,<:AbstractNode{D},T,<:AbstractEncoder{T},L}, X::AbstractMatrix{T}) where {D,T,L}
    out = MMatrix{D,size(X, 2),Float64}(undef)

    for (i, x) in zip(axes(out, 2), eachcol(X))
        out[:, i] = predict(d, x)
    end

    out
end

function reset!(d::Discriminator; seed::Union{Nothing,Int}=nothing)
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
function Base.show(io::IO, d::Discriminator{D,RegressionNode{D},T,E,L}) where {D,T,E,L}
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

function train!(d::Discriminator{D,RegressionNode{D},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}, y::StaticArray{Tuple{D},Float64,1}) where {D,T,L}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y)
    end

    return nothing
end

# TODO: Have γ be a parameter and make a specialized version for the case γ = 1.0
function predict(d::Discriminator{D,RegressionNode{D},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}) where {D,T,L}
    running_numerator = zero(MVector{D,Float64})
    running_denominator = zero(MVector{D,Float64})

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        el = predict(node, key)

        running_numerator .+= el.value

        if node.γ != 1.0
            running_denominator .+= @. (1 - node.γ^el.count) / (1 - node.γ)
        else
            running_denominator .+= el.count
        end
    end

    return running_denominator == 0 ? zero(MVector{D,Float64}) : running_numerator ./ running_denominator
end

# ------------------------- Functional Discriminator ------------------------- #
function Base.show(io::IO, d::Discriminator{D,FunctionalNode{D},T,E,L}) where {D,T,E,L}
    print(
        io,
        """Functional Discriminator
        ├ Input length: $(d.input_len)
        ├ Tuple size: $(d.n)
        ├ η: $(d.η)
        ├ Encoder type: $(E)
        ├ Partitioning scheme: $(d.partitioner)
        └ Loss function: $(L)"""
    )
end

function train!(d::Discriminator{D,FunctionalNode{D},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}, y::StaticArray{Tuple{D},Float64,1}) where {D,T,L}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    y_pred = predict(d, x)
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), -d.η * grad(d.loss, y, y_pred))
    end

    return nothing
end

function predict(d::Discriminator{D,FunctionalNode{D},T,<:AbstractEncoder{T},L}, x::AbstractVector{T}) where {D,T,L}
    cum_weight = zero(MVector{D,Float64})

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        weight = predict(node, key)

        cum_weight += weight
    end

    return cum_weight ./ length(d.nodes)
end


end

