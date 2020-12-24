using .Nodes
using ..Partitioners

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

RegressionDiscriminator = Discriminator{RandomPartitioner,RegressionNode}
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

# TODO: These predict operations might not be type-stable when the node type is
#       AccNode: They might return booleans and vectors of booleans or integers
#       and vectors of integers, respectively
#       It turns out it is type-stable because I initialize response with a Int
#       and vector of Ints, respectively 
# TODO: Output response in relative terms? (i.e. divide by the number of nodes)
# TODO: Can I merge this two functions into a single generic one? Most of the
#       body is the same.
function predict(d::Discriminator, X::T) where {T <: AbstractVector{Bool}}
    response = zero(Int)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += Nodes.predict(node, x)
    end

    return response
end

function predict(d::Discriminator, X::T) where {T <: AbstractMatrix{Bool}}
    response = zeros(Int, size(X, 1))

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += Nodes.predict(node, x)
    end

    return response
end

# I should be able to avoid all this replication. Should I just make the 
# bleaching threshold available to all discriminator types? The problem is that
# with discriminators that use DictNode, any value for b other than 0 doesn't
# make sense
function predict(d::BleachingDiscriminator, X::T; b=0) where {T <: AbstractVector{Bool}}
    response = zero(Int)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += Nodes.predict(node, x; b)
    end

    return response
end

function predict(d::BleachingDiscriminator, X::T; b=0) where {T <: AbstractMatrix{Bool}}
    response = zeros(Int, size(X, 1))

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += Nodes.predict(node, x; b)
    end

    return response
end

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

function predict(d::Discriminator{P,RegressionNode}, X::AbstractMatrix{Bool}) where {P <: AbstractPartitioner}
    [predict(d, x) for x in eachrow(X)]
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
            # α = 1 / count
            running_numerator += estimate / α
            running_denominator += 1 / α
        end
    end

    return running_denominator == 0 ? zero(Float64) : running_numerator / running_denominator
end

function predict(d::Discriminator{P,GeneralizedRegressionNode}, X::AbstractMatrix{Bool}) where {P <: AbstractPartitioner}
    [predict(d, x) for x in eachrow(X)]
end
