module ramnet

using Random

export Discriminator

include("Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper

struct Discriminator
    address_size::Int
    nodes::Vector{Dict{BitVector,Int8}}

    function Discriminator(X::AbstractVector{<:AbstractVector{Bool}}, n::Int)
        n < 1 && throw(ArgumentError("Address size must not be less then 1"))

        # Create the list of nodes
        # In order to do that, I need to know the number of partitions
    end
end

# Discriminator() = Discriminator(Vector())

# function train!(disc::Discriminator, )

# end


end
