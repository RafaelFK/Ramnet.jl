module ramnet

include("Utils.jl")

include("Encoders/Encoders.jl")

include("Partitioners/Partitioners.jl")

using ramnet.Partitioners
export partition, RandomPartitioner, random_partitioning

include("Models/Models.jl")
using ramnet.Models
export train!, predict

export Discriminator, BitDiscriminator, RegressionDiscriminator, GeneralizedRegressionDiscriminator

export MultiDiscriminatorClassifier

end
