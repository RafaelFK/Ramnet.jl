using .Nodes
using ..Partitioners

using Base.Iterators: zip

# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data) 
struct Discriminator{P <: AbstractPartitioner,T <: AbstractNode} <: AbstractModel
    partitioner::P
    nodes::Vector{T}
end

function Discriminator{P,T}(partitioner::P; kargs...) where {P <: AbstractPartitioner,T <: AbstractNode}
    nodes = [T(;kargs...) for _ in 1:length(partitioner)]
    
    Discriminator{P,T}(partitioner, nodes)
end

function Discriminator{RandomPartitioner,T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode}
    partitioner = RandomPartitioner(width, n; seed)

    Discriminator{RandomPartitioner,T}(partitioner; kargs...)
end

function Discriminator{P,T}(X::U, partitioner::P; kargs...) where {P <: AbstractPartitioner,T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{P,T}(partitioner; kargs...)

    train!(d, X)

    return d
end

function Discriminator{RandomPartitioner,T}(X::U, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{RandomPartitioner,T}(size(X)[end], n; seed, kargs...)

    train!(d, X)

    return d
end

## Aliases and convenience constructors
# Default node is DictNode with random partitioning
Discriminator(args...; kargs...) = Discriminator{RandomPartitioner,DictNode}(args...; kargs...)

const BitDiscriminator        = Discriminator{RandomPartitioner,DictNode{BitVector}}
const BleachingDiscriminator  = Discriminator{RandomPartitioner,AccNode} # This should be the default classification discriminator

# RegressionDiscriminator = Discriminator{RandomPartitioner,RegressionNode}
const RegressionDiscriminator = Discriminator{RandomPartitioner,FastRegressionNode}
const FastRegressionDiscriminator = Discriminator{RandomPartitioner,FastRegressionNode}
GeneralizedRegressionDiscriminator = Discriminator{RandomPartitioner,GeneralizedRegressionNode}

# Generic functions
function train!(d::Discriminator{<:AbstractPartitioner,<:AbstractClassificationNode}, X::T) where {T <: AbstractVecOrMat{Bool}}
    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        Nodes.train!(node, x)
    end

    return nothing
end

function train!(d::Discriminator{<:AbstractPartitioner,<:AbstractRegressionNode}, X, y)
    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        Nodes.train!(node, x, y)
    end

    return nothing
end

initial_response(::AbstractVector{Bool}) = zero(Int)
initial_response(X::AbstractMatrix{Bool}) = zeros(Int, size(X, 1))

# TODO: Output response in relative terms? (i.e. divide by the number of nodes)
function predict(d::Discriminator{<:AbstractPartitioner,<:AbstractClassificationNode}, X::AbstractVecOrMat{Bool}; b::Int=0)
    response = initial_response(X)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += predict(node, x; b)
    end

    return response
end

function predict(d::Discriminator{<:AbstractPartitioner,<:AbstractRegressionNode}, X::AbstractMatrix{Bool})
    [predict(d, x) for x in eachrow(X)]
end

# TODO: In the context of discriminators, predict and predict_response are the same thing.
#       Implement predict_response as an alias to predict

################################################################################
# Specialized training and prediction methods for the regresion variant

function predict(d::Discriminator{P,RegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, sum = predict(node, x)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
        end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

################################################################################
# Specialized training and prediction methods for the generalized regresion discriminator

# This expects that α is a constant
function predict(d::Discriminator{P,GeneralizedRegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    running_numerator = zero(Float64)
    running_denominator = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, estimate = predict(node, x)

        α = node.α
        if count != 0
            running_numerator += estimate / α
            running_denominator += 1 / α
        end
    end

    return running_denominator == 0 ? zero(Float64) : running_numerator / running_denominator
end

################################################################################
# Specialized training and prediction methods for the fast regresion variant

function predict(d::Discriminator{P,FastRegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, sum = predict(node, x)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
        end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

# =============================== Experimental =============================== #
using ..Encoders

# TODO: If the partitioning is going to be done upfront and from that moment
#       onwards each node knows how to form its tuples, maybe it's not necessary
#       for the partitioner to be a member of the Discriminator struct
struct AltDiscriminator{T <: Real,E <: AbstractEncoder{T}}
    input_len::Int
    encoder::E
    tuples::Vector{Vector{Int}}
    nodes::Vector{AltRegressionNode}
end

function AltDiscriminator(input_len::Int, n::Int, encoder::E; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64

    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    tuples = random_tuples(input_len * resolution(encoder), n; seed)
    nodes = [AltRegressionNode(;kargs...) for _ in 1:length(tuples)]

    AltDiscriminator{T,E}(input_len, encoder, tuples, nodes)
end

function train!(d::AltDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {T <: Real}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (t, node) in Iterators.zip(d.tuples, d.nodes)
        train!(node, encode(d.encoder, x, t), y)
    end

    return nothing
end

function train!(d::AltDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {T <: Real}
    size(X, 1) != d.input_len && throw(DimensionMismatch("expected number of rows of X to be to be $(d.input_len). Got $(size(X, 1))"))
    size(X, 2) != length(y) && throw(DimensionMismatch("The number of columns of X must match the length of y"))
    
    for i in eachindex(y)
        train!(d, X[:, i], y[i])
    end

    nothing
end

function predict(d::AltDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {T <: Real}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (t, node) in Iterators.zip(d.tuples, d.nodes)
        key = encode(d.encoder, x, t)
        count, sum = predict(node, key)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
        end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

function predict(d::AltDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}) where {T <: Real}
    [predict(d, x) for x in eachcol(X)]
end

# ------------------------------------------------------------------------------

struct SuperAltDiscriminator{T <: Real,E <: AbstractEncoder{T}}
    input_len::Int
    n::Int
    encoder::E
    # tuples::Vector{Vector{Tuple{Int,Int}}}
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{AltRegressionNode}
end

# TODO: Leia isso aqui!
#       Lidar com vetores de vetores de tuplas é desnecessariamente complicado. Simplesmente
#       mantenha segmentos e offsets em vetores separados. Também considere usar vetores "planos"
#       ao invés de vetores de vetores. Posso gerar os grupinhos de n elementos usando
#       Iterators.partition
function SuperAltDiscriminator(input_len::Int, n::Int, encoder::E; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)
    segments, offsets = random_tuples_segment_offset(input_len, res; seed)

    nodes = [AltRegressionNode(;kargs...) for _ in 1:cld(input_len * res, n)]

    SuperAltDiscriminator{T,E}(input_len, n, encoder, segments, offsets, nodes)
end

function train!(d::SuperAltDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {T <: Real}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y)
    end

    return nothing
end

function train!(d::SuperAltDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {T <: Real}
    size(X, 1) != d.input_len && throw(DimensionMismatch("expected number of rows of X to be to be $(d.input_len). Got $(size(X, 1))"))
    size(X, 2) != length(y) && throw(DimensionMismatch("The number of columns of X must match the length of y"))
    
    for i in eachindex(y)
        train!(d, X[:, i], y[i])
    end

    nothing
end

function predict(d::SuperAltDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {T <: Real}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        count, sum = predict(node, key)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
        end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

function predict(d::SuperAltDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}) where {T <: Real}
    [predict(d, x) for x in eachcol(X)]
end
