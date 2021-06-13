module Ramnet

function reset! end

include("Utils.jl")

include("Encoders/Encoders.jl")

include("Partitioners/Partitioners.jl")

using Ramnet.Partitioners
export partition, RandomPartitioner, random_partitioning

include("Models/Models.jl")
using Ramnet.Models
export train!, predict

export Discriminator,
  BitDiscriminator,
  RegressionDiscriminator,
  FastRegressionDiscriminator,
  GeneralizedRegressionDiscriminator,
  AltDiscriminator,
  SuperAltDiscriminator,
  DifferentialDiscriminator,
  kernel, kernel_weight, mix_kernels,
  FunctionalDiscriminator,
  add_kernel!,
  MultiFunctionalDiscriminator,
  min_mse_loss

export MultiDiscriminatorClassifier

include("Loss/Loss.jl")

include("Optimizers/Optimizers.jl")

end
