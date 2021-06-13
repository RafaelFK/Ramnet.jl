module Optimizers

using ..Loss
using ..Models.AltDiscriminators:AdaptiveFunctionalDiscriminator, predict

import ..reset!

export FunctionalOptimizer, AdaptiveOptimizer, learning_rate, grad, max_epochs

abstract type AbstractOptimizer end

mutable struct FunctionalOptimizer{L <: AbstractLoss} <: AbstractOptimizer
    loss::L
    start_learning_rate::Float64
    end_learning_rate::Float64
    learning_rate_decay::Int
    train_count::Int
    epochs::Int
end

# function FunctionalOptimizer(loss::L; learning_rate=0.01, epochs=1) where L
#     FunctionalOptimizer{L}(loss, learning_rate, learning_rate, 0, -1, epochs)
# end

function FunctionalOptimizer(loss::L; start_learning_rate, end_learning_rate, learning_rate_decay, epochs=1) where L
    FunctionalOptimizer{L}(loss, start_learning_rate, end_learning_rate, learning_rate_decay, -1, epochs)
end

# learning_rate(opt::FunctionalOptimizer{L}) where L = opt.learning_rate

function learning_rate(opt::FunctionalOptimizer)
    α = opt.train_count / opt.learning_rate_decay
    (1 - α) * opt.start_learning_rate + α * opt.end_learning_rate
end

# grad(opt::FunctionalOptimizer{L}, args...) where L = Loss.grad(opt.loss, args...)

function grad(opt::FunctionalOptimizer, args...)
    g = Loss.grad(opt.loss, args...)
    opt.train_count = opt.train_count == opt.learning_rate_decay ? opt.learning_rate_decay : opt.train_count + 1
        
    return g
end

function reset!(opt::FunctionalOptimizer)
    opt.train_count = -1

    nothing
end

max_epochs(opt::FunctionalOptimizer{L}) where L = opt.epochs


struct AdaptiveOptimizer{D,L <: AbstractLoss} <: AbstractOptimizer
    loss::L
    λ::Float64
    meta_rate::Float64
    learning_rate::Vector{Float64}
    grad_trace::Vector{Float64}
end

function AdaptiveOptimizer{D}(loss::L; λ::Float64,  meta_rate::Float64, initial_rate::Float64) where {D,L <: AbstractLoss}
    AdaptiveOptimizer{D,L}(loss, λ, meta_rate, fill(initial_rate, D), zeros(Float64, D))
end

learning_rate(opt::AdaptiveOptimizer) = opt.learning_rate

function update_learning_rate!(opt::AdaptiveOptimizer, gradient, trace)
    @. opt.learning_rate *= max(0.5, 1 - opt.meta_rate * gradient * trace)

    return nothing
end

function grad(opt::AdaptiveOptimizer{D,L}, f::AdaptiveFunctionalDiscriminator{D,T,E}, x, y_true) where {D,L,T,E}
    y_pred, trace = predict(f, x)
    gradient = Loss.grad(opt.loss, y_true, y_pred)

    update_learning_rate!(opt, gradient, trace)

    trace_update = gradient .+ opt.λ * hessian(opt.loss, y_true, y_pred) * trace

    return gradient, trace_update
end

end