module AltNodes

using StaticArrays

import ..AbstractModel, ..train!, ..predict, ..reset!

export AbstractNode, RegressionNode, FunctionalNode

abstract type AbstractNode{D} <: AbstractModel end

## Generic functions
reset!(node::N) where {N <: AbstractNode} = empty!(node.memory)

predict(node::N, X::UInt) where {N <: AbstractNode} = get(node.memory, X, elzero(N))

# ============================== Concrete types ============================== #
# ------------------------------ Regression Node ----------------------------- #
struct RegressionNodeMemoryElement{D}
    count::Vector{Int}
    value::Vector{Float64}

    function RegressionNodeMemoryElement{D}() where {D}
        new(zeros(Int, D), zeros(Float64, D))
    end
end


struct RegressionNode{D} <: AbstractNode{D}
    γ::Float64
    memory::Dict{UInt,RegressionNodeMemoryElement{D}}
end

function RegressionNode{D}(; γ=1.0) where {D}
    RegressionNode{D}(
        γ,
        Dict{UInt,RegressionNodeMemoryElement{D}}()
        )
end
    
elzero(::Type{RegressionNode{D}}) where {D} = RegressionNodeMemoryElement{D}()
# Base.eltype(::Type{RegressionNode{D}}) where {D} = RegressionNodeMemoryElement{D}

function train!(node::RegressionNode{D}, X::UInt, y::StaticArray{Tuple{D},Float64,1}) where {D}
    el = get!(node.memory, X, elzero(RegressionNode{D}))

    el.count .+= 1
    el.value .*= node.γ
    el.value .+= y 

    nothing
end

# ------------------------------ Functional Node ----------------------------- #
struct FunctionalNode{D} <: AbstractNode{D}
    memory::Dict{UInt,Vector{Float64}}
end

function FunctionalNode{D}() where {D}
    FunctionalNode{D}(Dict{UInt,Vector{Float64}}())
end

elzero(::Type{FunctionalNode{D}}) where D = zeros(Float64, D)
# Base.eltype(::Type{FunctionalNode{D}}) where {D} = Vector{Float64}

function train!(node::FunctionalNode{D}, X::UInt, y::StaticArray{Tuple{D},Float64,1}) where {D}
    value = get!(node.memory, X, elzero(FunctionalNode{D}))

    value .+= y

    nothing
end

end