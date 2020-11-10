module ramnet

using Random

export Discriminator

include("Mappers/Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper, random_mapping

struct Discriminator
    address_size::Int
    mapper::RandomMapper
    nodes::Vector{Dict{<:AbstractVector{Bool},Int8}}

    # It's better to have this type as a parameter of the struct
    function Discriminator(input_type::Type{<:AbstractVector{Bool}}, width::Int, n::Int; seed::Union{Nothing,Int}=nothing)
        # Create mapper
        mapper = RandomMapper(width, n; seed)

        # Create nodes
        nodes = [Dict{input_type,Int8}() for _ in 1:length(mapper)]

        new(n, mapper, nodes)
    end
end

# Discriminator() = Discriminator(Vector())

# function train!(disc::Discriminator, )

# end


end
