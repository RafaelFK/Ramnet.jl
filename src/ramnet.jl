module ramnet

# TODO: Should this be here?
using Random

include("Utils.jl")

include("Mappers/Mappers.jl")

using ramnet.Mappers
export RandomMapper, random_mapping

include("Nodes.jl")

include("Models/Models.jl")
using ramnet.Models
export Discriminator, StandardDiscriminator, BitDiscriminator
export train!, predict

export MultiDiscriminatorClassifier
export train!, predict, predict_response

end
