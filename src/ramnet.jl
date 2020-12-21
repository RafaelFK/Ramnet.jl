module ramnet

include("Utils.jl")

include("Encoders/Encoders.jl")

include("Mappers/Mappers.jl")

using ramnet.Mappers
export RandomMapper, random_mapping

include("Models/Models.jl")
using ramnet.Models
export train!, predict

export Discriminator, BitDiscriminator, RegressionDiscriminator, GeneralizedRegressionDiscriminator

export MultiDiscriminatorClassifier

end
