module ramnet

using Random

include("Mappers/Mappers.jl")

# Reexporting mappers for convenience
using ramnet.Mappers
export RandomMapper, random_mapping

include("Nodes.jl")

include("Models/Models.jl")
using ramnet.Models
export Discriminator, StandardDiscriminator, BitDiscriminator
export train!, predict

end
