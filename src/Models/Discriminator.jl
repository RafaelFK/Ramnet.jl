using .Nodes
using ..Mappers

# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data) 
struct Discriminator{T <: AbstractNode} <: AbstractModel
    mapper::RandomMapper
    nodes::Vector{T}

    function Discriminator{T}(mapper::RandomMapper) where {T <: AbstractNode}
        nodes = [T() for _ in 1:length(mapper)]
        
        new{T}(mapper, nodes)
    end
end

function Discriminator{T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractNode}
    mapper = RandomMapper(width, n; seed)

    Discriminator{T}(mapper)
end

function Discriminator{T}(X::U, mapper::RandomMapper) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{T}(mapper)

    train!(d, X)

    return d
end

function Discriminator{T}(X::U, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{T}(size(X)[end], n; seed)

    train!(d, X)

    return d
end

# Default node is DictNode
Discriminator(args...; kargs...) = Discriminator{DictNode}(args...; kargs...)

# const StandardDiscriminator  = Discriminator{DictNode{Vector{Bool}}}
const BitDiscriminator       = Discriminator{DictNode{BitVector}}
const BleachingDiscriminator = Discriminator{AccNode}

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

# I should be able to avoid all this replication
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