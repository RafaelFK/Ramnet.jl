module Nodes

import ..AbstractModel, ..train!, ..predict

export AbstractNode, DictNode, AccNode, RegressionNode

# TODO: There is nothing preventing different sized inputs
abstract type AbstractNode <: AbstractModel end

# If both BitVectors and BoolVectors hash to the same value when they have the same
# content, does it matter which type I choose as key for the underlying dictionary?
# I should check if both DictNode{BitVector} and DictNode{Vector{Bool}} accept any
# instance of AbstractVector{Bool}. If that's the case, I should get rid of the 
# type parameter and enforce the use of one of the types. Does it matter which one
# I choose? Is there an impact in performance and/or memory usage?
struct DictNode{K <: AbstractVector{Bool}} <: AbstractNode
    dict::Dict{K,Int8}
end

DictNode{T}() where {T <: AbstractVector{Bool}} = DictNode{T}(Dict{T,Int8}())

DictNode() = DictNode{Vector{Bool}}()

function train!(node::DictNode, X::T) where {T <: AbstractVector{Bool}}
    node.dict[X] = 1
end

function train!(node::DictNode, X::T) where {T <: AbstractMatrix{Bool}}
    for x in eachrow(X)
        node.dict[x] = 1
    end
end

# Would it make sense for the prediction to be binary as well?
function predict(node::DictNode, X::T) where {T <: AbstractVector{Bool}}
    return get(node.dict, X, zero(Int8))
end

function predict(node::DictNode, X::T) where {T <: AbstractMatrix{Bool}}
    return [get(node.dict, x, zero(Int8)) for x in eachrow(X)]
end


################################################################################
struct AccNode <: AbstractNode
    acc::Dict{Vector{Bool},Int64}
end

AccNode() = AccNode(Dict{Vector{Bool},Int64}())

function train!(node::AccNode, X::T) where {T <: AbstractVector{Bool}}
    node.acc[X] = get(node.acc, X, 0) + 1

    nothing
end

function train!(node::AccNode, X::T) where {T <: AbstractMatrix{Bool}}
    for x in eachrow(X)
        train!(node, x)
    end
end

function predict(node::AccNode, X::T; b::Int=0) where {T <: AbstractVector{Bool}}
    get(node.acc, X, 0) > b
end

function predict(node::AccNode, X::T; b::Int=0) where {T <: AbstractMatrix{Bool}}
    return [predict(node, x; b) for x in eachrow(X)]
end


################################################################################
struct RegressionNode{T <: Real} <: AbstractNode
    γ::Float64
    dict::Dict{Vector{Bool},Tuple{Int,T}}

    function RegressionNode{T}(γ, dict) where {T <: Real}
        if !(0.0 ≤ γ ≤ 1.0)
            throw(DomainError(γ, "`γ` must lie in the [0, 1] interval"))
        end

        new{T}(γ, dict)
    end
end

RegressionNode{T}(;γ=1.0) where {T <: Real} = RegressionNode{T}(γ, Dict{Vector{Bool},Tuple{Int,T}}())
RegressionNode(;γ=1.0) = RegressionNode{Float64}(γ)

function train!(node::RegressionNode{S}, X::T, y::S) where {S <: Real,T <: AbstractVector{Bool}}
    count, sum = get(node.dict, X, (zero(Int), zero(S)))
    
    node.dict[X] = (count + 1, node.γ * sum + y)

    nothing
end

function train!(node::RegressionNode{S}, X::T, y::AbstractVector{S}) where {S <: Real,T <: AbstractMatrix{Bool}}
    for (x, target) in Iterators.zip(eachrow(X), y)
        train!(node, x, target)
    end
end

function predict(node::RegressionNode{S}, X::T) where {S <: Real,T <: AbstractVector{Bool}}
    count, sum = get(node.dict, X, (zero(Int), zero(S)))

    if count == 0
        denominator = 0
    elseif node.γ != 1.0
        denominator = (1 - node.γ^count) / (1 - node.γ)
    else
        denominator = count
    end

    # return count == 0 ? (0, 0) : (count, sum)
    return (denominator, sum)
end

function predict(node::RegressionNode{S}, X::T) where {S <: Real,T <: AbstractMatrix{Bool}}
    return [predict(node, x) for (x, d) in eachrow(X)]
end

end