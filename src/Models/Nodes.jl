module Nodes

import ..AbstractModel, ..train!, ..predict

export AbstractNode,
    AbstractClassificationNode,
    AbstractRegressionNode,
    DictNode, 
    AccNode,
    RegressionNode,
    FastRegressionNode,
    GeneralizedRegressionNode,
    AltRegressionNode,
    stepsize,
    DifferentialNode,
    FunctionalNode

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

################################################################################
struct FastRegressionNode <: AbstractRegressionNode
    γ::Float64
    default::Tuple{Int,Float64}
    memory::Dict{UInt64,Tuple{Int,Float64}}

    function FastRegressionNode(γ, default, memory)
        if !(0.0 ≤ γ ≤ 1.0)
            throw(DomainError(γ, "`γ` must lie in the [0, 1] interval"))
        end

        new(γ, default, memory)
    end
end

FastRegressionNode(;γ=1.0, default=(zero(Int), zero(Float64))) = FastRegressionNode(γ, default, Dict{UInt64,Tuple{Int,Float64}}())

function predict(node::FastRegressionNode, X::T) where {T <: AbstractVector{Bool}}
    return get(node.memory, BitVector(X).chunks[1], node.default)
end

function train!(node::FastRegressionNode, X::T, y::Float64) where {T <: AbstractVector{Bool}}
    key = BitVector(X).chunks[1]
    count, sum = get(node.memory, key, node.default)
    
    node.memory[key] = (count + 1, node.γ * sum + y)

    nothing
end

# =============================== Experimental =============================== #
struct AltRegressionNode{S} <: AbstractRegressionNode
    α::Float64
    default::Tuple{Int,Float64}
    memory::Dict{UInt,Tuple{Int,Float64}}
    
    function AltRegressionNode{S}(α, default, memory) where {S}
        if !(0.0 ≤ α ≤ 1.0)
            throw(DomainError(α, "`α` must lie in the [0, 1] interval"))
        end
        
        new(α, default, memory)
    end
end

# TODO: Rename :original regression style to :average
AltRegressionNode(;style=:original, α=1.0, default=(zero(Int), zero(Float64))) = AltRegressionNode{style}(α, default, Dict{UInt64,Tuple{Int,Float64}}())

stepsize(node::AltRegressionNode{S}, count::Int) where {S} = stepsize(Val(S), node.α, count)

# Count cannot be zero
stepsize(::Val{:original}, ::Float64, count::Int) = count == 0 ? one(Float64) : one(Float64) / count
stepsize(::Val{:constant}, α::Float64, ::Int) = α
# There are two edge cases that I have to consider here: α = 1 and count = 0
# The equation bellow probably is not valid for these cases
function stepsize(::Val{:discounted}, α::Float64, count::Int)
    if α == 1.0
        return stepsize(Val(:original), α, count)
    elseif count == 0
        return 1.0
    else
        return (1 - α) / (1 - α^count)
    end
end

function predict(node::AltRegressionNode, X::UInt)
    return get(node.memory, X, node.default)
end

function train!(node::AltRegressionNode, X::UInt, y::Float64)
    count, estimate = get(node.memory, X, node.default)
    
    node.memory[X] = (count + 1, estimate + stepsize(node, count) * (y - estimate))

    nothing
end

# function train_mse!(node::AltRegressionNode, X::UInt, y::float64)
#     nothing
# end

# Specialization for the generic methods

function predict(node::AltRegressionNode, X::Vector{UInt}; kargs...)
    return [predict(node, x; kargs...) for x in X]
end


function train!(node::AltRegressionNode, X::Vector{UInt}, y::AbstractVector{Float64})
    for (x, target) in Iterators.zip(X, y)
        train!(node, x, target)
    end
end

# ============================ Super Experimental ============================ #
struct DifferentialNode <: AbstractRegressionNode
    α::Float64
    default::Float64
    memory::Dict{UInt,Float64}
    
    function DifferentialNode(α, default, memory)
        if !(0.0 ≤ α ≤ 1.0)
            throw(DomainError(α, "`α` must lie in the [0, 1] interval"))
        end
        
        new(α, default, memory)
    end
end

DifferentialNode(;α=1.0, default=zero(Float64)) = DifferentialNode(α, default, Dict{UInt64,Float64}())

function predict(node::DifferentialNode, X::UInt)
    return get(node.memory, X, node.default)
end

function train!(node::DifferentialNode, X::UInt, y::Float64, ŷ::Float64)
    weight = get(node.memory, X, node.default)
    
    node.memory[X] = weight + node.α * (y - ŷ)

    nothing
end

function predict(node::DifferentialNode, X::Vector{UInt}; kargs...)
    return [predict(node, x; kargs...) for x in X]
end

# This one is not quite correct
function train!(node::DifferentialNode, X::Vector{UInt}, y::AbstractVector{Float64})
    for (x, target) in Iterators.zip(X, y)
        train!(node, x, target)
    end
end

# =========================== Functional Gradient ============================ #
struct FunctionalNode <: AbstractRegressionNode
    memory::Dict{UInt,Float64}
end

FunctionalNode() = FunctionalNode(Dict{UInt,Float64}())

function predict(node::FunctionalNode, X::UInt)
    return get(node.memory, X, zero(Float64))
end

function train!(node::FunctionalNode, X::UInt, y::Float64)
    weight = get(node.memory, X, zero(Float64))
    
    node.memory[X] = weight + y

    return nothing
end

reset!(node::FunctionalNode) = empty!(node.memory)

end