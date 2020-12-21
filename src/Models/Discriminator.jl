using .Nodes
using ..Mappers

# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data) 
struct Discriminator{T <: AbstractNode} <: AbstractModel
    mapper::RandomMapper
    nodes::Vector{T}
end

function Discriminator{T}(mapper::Union{RandomMapper,Nothing}=nothing; kargs...) where {T <: AbstractNode}
    nodes = [T(;kargs...) for _ in 1:length(mapper)]
    
    Discriminator{T}(mapper, nodes)
end

function Discriminator{T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode}
    mapper = RandomMapper(width, n; seed)

    Discriminator{T}(mapper; kargs...)
end

function Discriminator{T}(X::U, mapper::RandomMapper; kargs...) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{T}(mapper; kargs...)

    train!(d, X)

    return d
end

function Discriminator{T}(X::U, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{T}(size(X)[end], n; seed, kargs...)

    train!(d, X)

    return d
end

# Default node is DictNode
Discriminator(args...; kargs...) = Discriminator{DictNode}(args...; kargs...)

# const StandardDiscriminator  = Discriminator{DictNode{Vector{Bool}}}
const BitDiscriminator        = Discriminator{DictNode{BitVector}}
const BleachingDiscriminator  = Discriminator{AccNode}
const RegressionDiscriminator = Discriminator{RegressionNode{Float64}}
const GeneralizedRegressionDiscriminator = Discriminator{GeneralizedRegressionNode}

function train!(d::Discriminator, X::T) where {T <: AbstractVecOrMat{Bool}}
    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        Nodes.train!(node, x)
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

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        response += Nodes.predict(node, x)
    end

    return response
end

function predict(d::Discriminator, X::T) where {T <: AbstractMatrix{Bool}}
    response = zeros(Int, size(X, 1))

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
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

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        response += Nodes.predict(node, x; b)
    end

    return response
end

function predict(d::BleachingDiscriminator, X::T; b=0) where {T <: AbstractMatrix{Bool}}
    response = zeros(Int, size(X, 1))

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        response += Nodes.predict(node, x; b)
    end

    return response
end

################################################################################
# Specialized training and prediction methods for the regresion variant

function train!(d::Discriminator{RegressionNode{S}}, X, y) where {S <: Real}
    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        Nodes.train!(node, x, y)
    end

    return nothing
end

function predict(d::Discriminator{RegressionNode{S}}, X::AbstractVector{Bool}) where {S <: Real}
    partial_count = zero(Int)
    estimate = zero(S)

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        count, sum = predict(node, x)

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

function predict(d::Discriminator{RegressionNode{S}}, X::AbstractMatrix{Bool}) where {S <: Real}
    [predict(d, x) for x in eachrow(X)]
end

################################################################################
# Specialized training and prediction methods for the generalized regresion discriminator

function train!(d::Discriminator{GeneralizedRegressionNode}, X, y)
    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        Nodes.train!(node, x, y)
    end

    return nothing
end

# This expects that α is a constant
function predict(d::Discriminator{GeneralizedRegressionNode}, X::AbstractVector{Bool})
    running_numerator = zero(Float64)
    running_denominator = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
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

function predict(d::Discriminator{GeneralizedRegressionNode}, X::AbstractMatrix{Bool})
    [predict(d, x) for x in eachrow(X)]
end

