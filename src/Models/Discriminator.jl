using ..Nodes
using ..Mappers

# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data) 
struct Discriminator{T <: AbstractVector{Bool}}
    # address_size::Int
    mapper::RandomMapper
    nodes::Vector{DictNode{T}}

    function Discriminator{T}(mapper::RandomMapper) where {T <: AbstractVector{Bool}}
        nodes = [DictNode{T}() for _ in 1:length(mapper)]
        
        new{T}(mapper, nodes)
    end
end

function Discriminator{T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractVector{Bool}}
    mapper = RandomMapper(width, n; seed)

    Discriminator{T}(mapper)
end

const StandardDiscriminator = Discriminator{Vector{Bool}}
const BitDiscriminator      = Discriminator{BitVector}

function train!(d::Discriminator, X::T) where {T <: VecOrMat{Bool}}
    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        Nodes.train!(node, x)
    end

    return nothing
end

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
