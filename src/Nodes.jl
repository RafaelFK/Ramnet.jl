module Nodes

abstract type Node end

struct DictNode{K <: AbstractVector{Bool}}
    dict::Dict{K,Int8}
end

DictNode(type::Type{T})

end