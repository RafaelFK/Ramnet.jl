module Models
    include("Discriminator.jl")
    export Discriminator, StandardDiscriminator, BitDiscriminator
    export train!, predict
end