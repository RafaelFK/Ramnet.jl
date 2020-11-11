module ramnet

using Random

include("Mappers/Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper, random_mapping

export Discriminator, StandardDiscriminator, BitDiscriminator
export train!, predict

struct Discriminator{T <: AbstractVector{Bool}}
    address_size::Int
    mapper::RandomMapper
    nodes::Vector{Dict{T,Int8}}

    # It's better to have this type as a parameter of the struct
    function Discriminator{T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: AbstractVector{Bool}}
        # Create mapper
        mapper = RandomMapper(width, n; seed)

        # Create nodes
        nodes = [Dict{T,Int8}() for _ in 1:length(mapper)]

        new{T}(n, mapper, nodes)
    end
end

const StandardDiscriminator = Discriminator{Vector{Bool}}
const BitDiscriminator      = Discriminator{BitVector}

function train_node!(node::Dict{<:AbstractVector{Bool},Int8}, X::T) where {T <: AbstractVector{Bool}}
    node[X] = 1

    return nothing
end

function train_node!(node::Dict{<: AbstractVector{Bool},Int8}, X::T) where {T <: AbstractMatrix{Bool}}
    for x in eachrow(X)
        node[x] = 1
    end

    return nothing
end

function train!(d::Discriminator, X::T) where {T <: VecOrMat{Bool}}
    # Train each node with their appropriate partition of the input
    # according to the mapper
    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        train_node!(node, x)
    end

    return nothing
end

function predict_node(node::Dict{<: AbstractVector{Bool},Int8}, X::T) where {T <: AbstractVector{Bool}}
    return get(node, X, zero(Int8))
end

function predict_node(node::Dict{<: AbstractVector{Bool},Int8}, X::T) where {T <: AbstractMatrix{Bool}}
    return [get(node, x, zero(Int8)) for x in eachrow(X)]
end

# TODO: Output response in relative terms? (i.e. divide by the number of nodes)
function predict(d::Discriminator, X::T) where {T <: AbstractVector{Bool}}
    response = zero(Int)

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        response += predict_node(node, x)
    end

    return response
end

function predict(d::Discriminator, X::T) where {T <: AbstractMatrix{Bool}}
    response = zeros(Int, size(X, 1))

    for (node, x) in Iterators.zip(d.nodes, map(d.mapper, X))
        response += predict_node(node, x)
    end

    return response
end

end
