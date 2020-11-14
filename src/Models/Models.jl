module Models

include("Discriminator.jl")
export Discriminator, StandardDiscriminator, BitDiscriminator
export train!, predict

include("MultiDiscriminatorClassifier.jl")
export MultiDiscriminatorClassifier
export train!, predict

end