module ramnet

using Random

include("Mappers/Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper, random_mapping

include("Nodes.jl")
import ramnet.Nodes
import ramnet.Nodes: DictNode

export Discriminator, StandardDiscriminator, BitDiscriminator
export train!, predict

struct Discriminator{T <: AbstractVector{Bool}}
    address_size::Int
    mapper::RandomMapper
    nodes::Vector{DictNode{T}}

    function Discriminator{T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractVector{Bool}}
        # Create mapper
        mapper = RandomMapper(width, n; seed)

        # Create nodes
        nodes = [DictNode{T}() for _ in 1:length(mapper)]

        new{T}(n, mapper, nodes)
    end
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

end
