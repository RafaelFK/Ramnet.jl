module Models

export train!,
  predict,
  predict_response,
  predict_bleached_response,
  predict_bleached

# A model (m) must implement the following functions:
#
# train!(m, X[, y]) - train model with pattern X and a label y with it's associative
# predict(m, X) - returns the response of the model for pattern X

abstract type AbstractModel end

function train! end
function predict end
function predict_response end
function predict_bleached_response end
function predict_bleached end

# I can provide a default implementation of train! and predict for matrix inputs,
# assuming that their vector counterparts have already been implemented somewhere
# else

include("Nodes.jl")
# export DictNode, AccNode

include("Discriminator.jl")
export Discriminator, BitDiscriminator, BleachingDiscriminator, RegressionDiscriminator

include("MultiDiscriminatorClassifier.jl")
export MultiDiscriminatorClassifier

end