module Nodes

export DictNode, train!, predict

abstract type Node end

# If both BitVectors and BoolVectors hash to the same value when they have the same
# content, does it matter which type I choose as key for the underlying dictionary?
# I should check if both DictNode{BitVector} and DictNode{Vector{Bool}} accept any
# instance of AbstractVector{Bool}. If that's the case, I should get rid of the 
# type parameter and enforce the use of one of the types. Does it matter which one
# I choose? Is there an impact in performance and/or memory usage?
struct DictNode{K <: AbstractVector{Bool}} <: Node
    dict::Dict{K,Int8}
end

DictNode{T}() where {T <: AbstractVector{Bool}} = DictNode{T}(Dict{T,Int8}())

function train!(node::DictNode, X::T) where {T <: AbstractVector{Bool}}
    node.dict[X] = 1
end

function train!(node::DictNode, X::T) where {T <: AbstractMatrix{Bool}}
    for x in eachrow(X)
        node.dict[x] = 1
    end
end

function predict(node::DictNode, X::T) where {T <: AbstractVector{Bool}}
    return get(node.dict, X, zero(Int8))
end

function predict(node::DictNode, X::T) where {T <: AbstractMatrix{Bool}}
    return [get(node.dict, x, zero(Int8)) for x in eachrow(X)]
end

end