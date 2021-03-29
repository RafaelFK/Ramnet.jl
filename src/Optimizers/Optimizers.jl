module Optimizers

using ..Loss

export FunctionalOptimizer, learning_rate, grad, max_epochs

abstract type AbstractOptimizer end

struct FunctionalOptimizer{L <: AbstractLoss} <: AbstractOptimizer
    loss::L
    learning_rate::Float64
    epochs::Int
end

function FunctionalOptimizer(loss::L; learning_rate=0.01, epochs=1) where L
    FunctionalOptimizer{L}(loss, learning_rate, epochs)
end

learning_rate(opt::FunctionalOptimizer{L}) where L = opt.learning_rate

grad(opt::FunctionalOptimizer{L}, args...) where L = Loss.grad(opt.loss, args...)

max_epochs(opt::FunctionalOptimizer{L}) where L = opt.epochs

end