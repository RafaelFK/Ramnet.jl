module Nodes

import ..AbstractModel, ..train!, ..predict

export AbstractNode,
    AbstractClassificationNode,
    AbstractRegressionNode,
    DictNode, 
    AccNode,
    RegressionNode,
    GeneralizedRegressionNode

# TODO: There is nothing preventing different sized inputs
abstract type AbstractNode <: AbstractModel end
abstract type AbstractClassificationNode <: AbstractNode end
abstract type AbstractRegressionNode <: AbstractNode end

# Generic methods

function train!(node::N, X::T) where {N <: AbstractClassificationNode,T <: AbstractMatrix{Bool}}
    for x in eachrow(X)
        train!(node, x)
    end
end

# TODO: This is implicitly requiring that a AbstractClassificationNode has a `memory` field
#       and a `default` field. It might be better to enforce this through traits.
function predict(node::N, X::T; b::Int=0) where {N <: AbstractClassificationNode,T <: AbstractVector{Bool}}
    return get(node.memory, X, node.default) > b
end

function predict(node::N, X::T; kargs...) where {N <: AbstractNode,T <: AbstractMatrix{Bool}}
    return [predict(node, x; kargs...) for x in eachrow(X)]
end


function train!(node::N, X::T, y::AbstractVector{Float64}) where {N <: AbstractRegressionNode,T <: AbstractMatrix{Bool}}
    for (x, target) in Iterators.zip(eachrow(X), y)
        train!(node, x, target)
    end
end

function predict(node::N, X::T) where {N <: AbstractRegressionNode,T <: AbstractVector{Bool}}
    return get(node.memory, X, node.default)
end

# If both BitVectors and BoolVectors hash to the same value when they have the same
# content, does it matter which type I choose as key for the underlying dictionary?
# I should check if both DictNode{BitVector} and DictNode{Vector{Bool}} accept any
# instance of AbstractVector{Bool}. If that's the case, I should get rid of the 
# type parameter and enforce the use of one of the types. Does it matter which one
# I choose? Is there an impact in performance and/or memory usage?
struct DictNode{K <: AbstractVector{Bool}} <: AbstractClassificationNode
    default::Int8
    memory::Dict{K,Int8}
end

DictNode{T}(;default=zero(Int8)) where {T <: AbstractVector{Bool}} = DictNode{T}(default, Dict{T,Int8}())

DictNode(;default=zero(Int8)) = DictNode{Vector{Bool}}(;default)

function train!(node::DictNode, X::T) where {T <: AbstractVector{Bool}}
    node.memory[X] = 1

    nothing
end

################################################################################
struct AccNode <: AbstractClassificationNode
    default::Int64
    memory::Dict{Vector{Bool},Int64}
end

AccNode(;default=zero(Int64)) = AccNode(default, Dict{Vector{Bool},Int64}())

function train!(node::AccNode, X::T) where {T <: AbstractVector{Bool}}
    node.memory[X] = get(node.memory, X, node.default) + 1

    nothing
end

################################################################################
struct RegressionNode <: AbstractRegressionNode
    γ::Float64
    default::Tuple{Int,Float64}
    memory::Dict{Vector{Bool},Tuple{Int,Float64}}

    function RegressionNode(γ, default, memory)
        if !(0.0 ≤ γ ≤ 1.0)
            throw(DomainError(γ, "`γ` must lie in the [0, 1] interval"))
        end

        new(γ, default, memory)
    end
end

RegressionNode(;γ=1.0, default=(zero(Int), zero(Float64))) = RegressionNode(γ, default, Dict{Vector{Bool},Tuple{Int,Float64}}())

function train!(node::RegressionNode, X::T, y::Float64) where {T <: AbstractVector{Bool}}
    count, sum = get(node.memory, X, node.default)
    
    node.memory[X] = (count + 1, node.γ * sum + y)

    nothing
end

################################################################################
# TODO: Enforce α to be greater than zero
struct GeneralizedRegressionNode <: AbstractRegressionNode
    α::Float64
    default::Tuple{Int,Float64}
    memory::Dict{Vector{Bool},Tuple{Int,Float64}}
end


GeneralizedRegressionNode(;α::Float64, default=(zero(Int), zero(Float64))) = GeneralizedRegressionNode(α, default, Dict{Vector{Bool},Tuple{Int,Float64}}())

# TODO: Make stepsize function that takes in a node and returns its appropriate α
function train!(node::GeneralizedRegressionNode, X::T, y::Float64) where {T <: AbstractVector{Bool}}
    count, estimate = get(node.memory, X, node.default)
    
    node.memory[X] = (count + 1, estimate + node.α * (y - estimate))

    nothing
end

end