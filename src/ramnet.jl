module ramnet

using Random

export Discriminator

include("Mappers/Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper, random_mapping

struct Discriminator{T <: VecOrMat{Bool}}
    address_size::Int
    mapper::RandomMapper
    nodes::Vector{Dict{AbstractVector{Bool},Int8}}

    function Discriminator(X::T, n::Int; seed::Union{Nothing,Int}=nothing) where {T <: VecOrMat{Bool}}
        n < 1 && throw(ArgumentError("Address size must not be less then 1"))

        # Create mapper
        # mapper = RandomMapper(X, n; seed)

        # Create nodes

    end
end

# Discriminator() = Discriminator(Vector())

# function train!(disc::Discriminator, )

# end


end
