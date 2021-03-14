module Loss

export AbstractLoss, NoneLoss, SquaredError, grad

abstract type AbstractLoss end

function grad(l::L, y_true, y_pred) where {L <: AbstractLoss}
    error("$(L): Unknown gradient")
end

struct NoneLoss <: AbstractLoss end

struct SquaredError <: AbstractLoss end

grad(::SquaredError, y_true, y_pred) = y_pred .- y_true

end