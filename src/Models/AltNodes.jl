module AltNodes

using StaticArrays

import ..AbstractModel, ..train!, ..predict, ..reset!

export AbstractNode, RegressionNode, FunctionalNode, RegularizedFunctionalNode, AdaptiveFunctionalNode

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

# ------------------------ Regularized Functional Node ----------------------- #
mutable struct RegularizedFunctionalNodeMemoryElement{D}
    count::UInt
    value::Vector{Float64}

    function RegularizedFunctionalNodeMemoryElement{D}() where {D}
        new(zero(UInt), zeros(Float64, D))
    end
end

struct RegularizedFunctionalNode{D} <: AbstractNode{D}
    α::Float64
    memory::Dict{UInt,RegularizedFunctionalNodeMemoryElement{D}}
end

function RegularizedFunctionalNode{D}(; α=1.0) where {D}
    RegularizedFunctionalNode{D}(
        α,
        Dict{UInt,RegularizedFunctionalNodeMemoryElement{D}}()
    )
end
    
elzero(::Type{RegularizedFunctionalNode{D}}) where {D} = RegularizedFunctionalNodeMemoryElement{D}()

function train!(node::RegularizedFunctionalNode{D}, X::UInt, y::StaticArray{Tuple{D},Float64,1}, global_count::UInt) where {D}
    el = predict(node, X, global_count)

    el.value .+= y
    node.memory[X] = el
    
    nothing
end

function predict(node::RegularizedFunctionalNode{D}, X::UInt, global_count::UInt) where {D}
    el = get(node.memory, X, RegularizedFunctionalNodeMemoryElement{D}())

    el.value .*= node.α^(global_count - el.count)
    el.count = global_count

    return el
end

# -------------------------- Adaptive Functional Node ------------------------ #
mutable struct AdaptiveFunctionalNodeMemoryElement{D}
    count::UInt
    value::Vector{Float64}

    function AdaptiveFunctionalNodeMemoryElement{D}() where {D}
        new(zero(UInt), zeros(Float64, 2 * D))
    end
end

struct AdaptiveFunctionalNode{D} <: AbstractNode{D}
    λ::Float64
    memory::Dict{UInt,AdaptiveFunctionalNodeMemoryElement{D}}
end

function AdaptiveFunctionalNode{D}(; λ=1.0) where {D}
    AdaptiveFunctionalNode{D}(
        λ,
        Dict{UInt,AdaptiveFunctionalNodeMemoryElement{D}}()
    )
end
    
elzero(::Type{AdaptiveFunctionalNode{D}}) where {D} = AdaptiveFunctionalNodeMemoryElement{D}()

function train!(node::AdaptiveFunctionalNode{D}, X::UInt, grad::StaticArray{Tuple{D},Float64,1}, v::StaticArray{Tuple{D},Float64,1}, global_count::UInt) where {D}
    el = predict(node, X, global_count)

    el.value[1:D] .+= grad
    el.value[D + 1:end] .+= v
    node.memory[X] = el
    
    nothing
end

function predict(node::AdaptiveFunctionalNode{D}, X::UInt, global_count::UInt) where {D}
    el = get(node.memory, X, AdaptiveFunctionalNodeMemoryElement{D}())

    el.value[D + 1:end] .*= node.λ^(global_count - el.count)
    el.count = global_count

    return el
end

end