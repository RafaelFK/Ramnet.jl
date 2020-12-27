module Ramnet

include("Utils.jl")

include("Encoders/Encoders.jl")

include("Partitioners/Partitioners.jl")

using Ramnet.Partitioners
export partition, RandomPartitioner, random_partitioning

include("Models/Models.jl")
using Ramnet.Models
export train!, predict

export Discriminator, BitDiscriminator, RegressionDiscriminator, GeneralizedRegressionDiscriminator

export MultiDiscriminatorClassifier

end
