module Models

import ..reset!

# using ..Loss
# using ..Optimizers

export train!,
  predict,
  predict_response

# A model (m) must implement the following functions:
#
# train!(m, X[, y]) - train model with pattern X and a label y with it's associative
# predict(m, X) - returns the response of the model for pattern X

abstract type AbstractModel end

function train! end
function predict end
function predict_response end

# I can provide a default implementation of train! and predict for matrix inputs,
# assuming that their vector counterparts have already been implemented somewhere
# else

include("Nodes.jl")

include("Discriminator.jl")
export Discriminator,
  BitDiscriminator,
  BleachingDiscriminator,
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

include("MultiDiscriminatorClassifier.jl")
export MultiDiscriminatorClassifier


include("AltNodes.jl")

include("AltDiscriminators.jl")

end
