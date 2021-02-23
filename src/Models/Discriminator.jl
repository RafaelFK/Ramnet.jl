using .Nodes
using ..Partitioners
using ..Encoders

using Base.Iterators:zip

# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data) 
struct Discriminator{P <: AbstractPartitioner,T <: AbstractNode} <: AbstractModel
    partitioner::P
    nodes::Vector{T}
end

function Discriminator{P,T}(partitioner::P; kargs...) where {P <: AbstractPartitioner,T <: AbstractNode}
    nodes = [T(;kargs...) for _ in 1:length(partitioner)]
    
    Discriminator{P,T}(partitioner, nodes)
end

function Discriminator{RandomPartitioner,T}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode}
    partitioner = RandomPartitioner(width, n; seed)

    Discriminator{RandomPartitioner,T}(partitioner; kargs...)
end

function Discriminator{P,T}(X::U, partitioner::P; kargs...) where {P <: AbstractPartitioner,T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{P,T}(partitioner; kargs...)

    train!(d, X)

    return d
end

function Discriminator{RandomPartitioner,T}(X::U, n::Int; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: AbstractNode,U <: AbstractVecOrMat{Bool}}
    d = Discriminator{RandomPartitioner,T}(size(X)[end], n; seed, kargs...)

    train!(d, X)

    return d
end

## Aliases and convenience constructors
# Default node is DictNode with random partitioning
Discriminator(args...; kargs...) = Discriminator{RandomPartitioner,DictNode}(args...; kargs...)

const BitDiscriminator        = Discriminator{RandomPartitioner,DictNode{BitVector}}
const BleachingDiscriminator  = Discriminator{RandomPartitioner,AccNode} # This should be the default classification discriminator

# RegressionDiscriminator = Discriminator{RandomPartitioner,RegressionNode}
# const RegressionDiscriminator = Discriminator{RandomPartitioner,FastRegressionNode}
const FastRegressionDiscriminator = Discriminator{RandomPartitioner,FastRegressionNode}
# GeneralizedRegressionDiscriminator = Discriminator{RandomPartitioner,GeneralizedRegressionNode}

# Generic functions
function train!(d::Discriminator{<:AbstractPartitioner,<:AbstractClassificationNode}, X::T) where {T <: AbstractVecOrMat{Bool}}
    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        Nodes.train!(node, x)
    end

    return nothing
end

function train!(d::Discriminator{<:AbstractPartitioner,<:AbstractRegressionNode}, X, y)
    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        Nodes.train!(node, x, y)
    end

    return nothing
end

initial_response(::AbstractVector{Bool}) = zero(Int)
initial_response(X::AbstractMatrix{Bool}) = zeros(Int, size(X, 1))

# TODO: Output response in relative terms? (i.e. divide by the number of nodes)
function predict(d::Discriminator{<:AbstractPartitioner,<:AbstractClassificationNode}, X::AbstractVecOrMat{Bool}; b::Int=0)
    response = initial_response(X)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        response += predict(node, x; b)
    end

    return response
end

function predict(d::Discriminator{<:AbstractPartitioner,<:AbstractRegressionNode}, X::AbstractMatrix{Bool})
    [predict(d, x) for x in eachrow(X)]
end

# TODO: In the context of discriminators, predict and predict_response are the same thing.
#       Implement predict_response as an alias to predict

################################################################################
# Specialized training and prediction methods for the regresion variant

function predict(d::Discriminator{P,RegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, sum = predict(node, x)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
        end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

################################################################################
# Specialized training and prediction methods for the generalized regresion discriminator

# This expects that α is a constant
function predict(d::Discriminator{P,GeneralizedRegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    running_numerator = zero(Float64)
    running_denominator = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, estimate = predict(node, x)

        α = node.α
        if count != 0
            running_numerator += estimate / α
            running_denominator += 1 / α
        end
    end

    return running_denominator == 0 ? zero(Float64) : running_numerator / running_denominator
end

################################################################################
# Specialized training and prediction methods for the fast regresion variant

function predict(d::Discriminator{P,FastRegressionNode}, X::AbstractVector{Bool}) where {P <: AbstractPartitioner}
    partial_count = zero(Int)
    estimate = zero(Float64)

    for (node, x) in Iterators.zip(d.nodes, partition(d.partitioner, X))
        count, sum = predict(node, x)

        if node.γ != 1.0
            count = (1 - node.γ^count) / (1 - node.γ)
    end

        partial_count += count
        estimate += sum
    end

    return partial_count == 0 ? estimate : estimate / partial_count
end

# =============================== Experimental =============================== #
# ---------------------------------------------------------------------------- #

struct SuperAltDiscriminator{S,T <: Real,E <: AbstractEncoder{T}}
    input_len::Int
    n::Int
    default::Tuple{Int,Float64}
    scaling::Float64
encoder::E
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{AltRegressionNode{S}}
end

# function SuperAltDiscriminator(input_len::Int, n::Int, encoder::E; style::Symbol=:original, seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
#     max_tuple_size = (UInt == UInt32) ? 32 : 64
#     (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

#     res = resolution(encoder)
#     segments, offsets = random_tuples_segment_offset(input_len, res; seed)

#     nodes = [AltRegressionNode(;style, kargs...) for _ in 1:cld(input_len * res, n)]

#     SuperAltDiscriminator{style,T,E}(input_len, n, encoder, segments, offsets, nodes)
# end

function SuperAltDiscriminator(input_len::Int, n::Int, encoder::E, partitioner::Function; style::Symbol=:original, seed::Union{Nothing,Int}=nothing, default::Tuple{Int,Float64}=(zero(Int), zero(Float64)), kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partitioner(input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [AltRegressionNode(;style, default, kargs...) for _ in 1:cld(input_len * res, n)]

    SuperAltDiscriminator{style,T,E}(input_len, n, default, 1.0, encoder, segments, offsets, nodes)
end

function SuperAltDiscriminator(input_len::Int, n::Int, encoder::E; partitioner::Symbol=:uniform, style::Symbol=:original, seed::Union{Nothing,Int}=nothing, default::Tuple{Int,Float64}=(zero(Int), zero(Float64)), kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    if partitioner == :uniform
        p_func = uniform_random_tuples
    elseif partitioner == :significance
        p_func = significance_aware_random_tuples
    else
        throw(DomainError(partitioner, "Unknown partitioning"))
    end

    SuperAltDiscriminator(input_len, n, encoder, p_func; style, seed, default, kargs...)
end

const RegressionDiscriminator = SuperAltDiscriminator

function train!(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {S,T <: Real}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y)
    end

    return nothing
end

function tuple_distance(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, z::AbstractVector{T}) where {S,T <: Real}
    dist = zero(Int)
    for (segments, offsets) in zip(Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        dist += Int(encode(d.encoder, x, segments, offsets) != encode(d.encoder, z, segments, offsets))
    end

    return dist
end

function kernel(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, z::AbstractVector{T}) where {S,T <: Real}
    return 1 - tuple_distance(d, x, z) / length(d.nodes)
end

function counter_sum(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {S,T <: Real}
    s = zero(Int)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        s += first(get(node.memory, encode(d.encoder, x, segments, offsets), node.default))
    end

    return s
end

function kernel_weight(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {S,T <: Real}
    s = counter_sum(d, x)

    return s == 0 ? zero(Float64) : length(d.nodes) * y / s
end

function mix_kernels(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, xs::AbstractMatrix{T}, ys::AbstractVector{Float64}) where {S,T <: Real}
    (x::AbstractVector{T}) -> sum([kernel_weight(d, x, y) * kernel(d, col, x) for (col, y) in Iterators.zip(eachcol(xs), ys)])
end

export kernel, kernel_weight, mix_kernels

function train_mse!(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {S,T <: Real}
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y)
    end

    return nothing
end

function train!(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, y::AbstractVector{Float64}) where {S,T <: Real}
    size(X, 1) != d.input_len && throw(DimensionMismatch("expected number of rows of X to be to be $(d.input_len). Got $(size(X, 1))"))
    size(X, 2) != length(y) && throw(DimensionMismatch("The number of columns of X must match the length of y"))
    
    for i in eachindex(y)
        train!(d, X[:, i], y[i])
    end

    nothing
end

function predict(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {S,T <: Real}
    running_numerator = zero(Float64)
    running_denominator = zero(Float64)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
            count, estimate = predict(node, key)

        if count != 0
            step = stepsize(node, count)
            running_numerator += estimate / step
            running_denominator += 1 / step
        end
    end

    # return running_denominator == 0 ? zero(Float64) : running_numerator / running_denominator
    return running_denominator == 0 ? last(d.default) : running_numerator / running_denominator
end

function predict(d::SuperAltDiscriminator{S,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}) where {S,T <: Real}
    [predict(d, x) for x in eachcol(X)]
end
    
# ============================ Super Experimental ============================ #

struct DifferentialDiscriminator{T <: Real,E <: AbstractEncoder{T}}
    input_len::Int
    n::Int
    encoder::E
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{DifferentialNode}
end

# function DifferentialDiscriminator(input_len::Int, n::Int, encoder::E; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
#     max_tuple_size = (UInt == UInt32) ? 32 : 64
#     (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

#     res = resolution(encoder)
#     segments, offsets = random_tuples_segment_offset(input_len, res; seed)

#     nodes = [DifferentialNode(; kargs...) for _ in 1:cld(input_len * res, n)]

#     DifferentialDiscriminator{T,E}(input_len, n, encoder, segments, offsets, nodes)
# end
    
function DifferentialDiscriminator(input_len::Int, n::Int, encoder::E, partitioner::Function; seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partitioner(input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [DifferentialNode(; kargs...) for _ in 1:cld(input_len * res, n)]

    DifferentialDiscriminator{T,E}(input_len, n, encoder, segments, offsets, nodes)
    end

function DifferentialDiscriminator(input_len::Int, n::Int, encoder::E; partitioner::Symbol=:uniform, seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    if partitioner == :uniform
        p_func = uniform_random_tuples
    elseif partitioner == :significance
        p_func = significance_aware_random_tuples
    else
        throw(DomainError(partitioner, "Unknown partitioning"))
    end

    DifferentialDiscriminator(input_len, n, encoder, p_func; seed)
end

# const RegressionDiscriminator = DifferentialDiscriminator
function predict(d::DifferentialDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {T <: Real}
    estimate = zero(Float64)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        weight = predict(node, key)

        estimate += weight
    end

    # return running_denominator == 0 ? zero(Float64) : running_numerator / running_denominator
    return estimate
end

function predict(d::DifferentialDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}) where {T <: Real}
    [predict(d, x) for x in eachcol(X)]
end

function train!(d::DifferentialDiscriminator{T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {T <: Real}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    ŷ = predict(d, x)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), y, ŷ)
    end

    return nothing
end

using Random

function train!(d::DifferentialDiscriminator{T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, y::AbstractVector{Float64}; epochs=1) where {T <: Real}
    size(X, 1) != d.input_len && throw(DimensionMismatch("expected number of rows of X to be to be $(d.input_len). Got $(size(X, 1))"))
    size(X, 2) != length(y) && throw(DimensionMismatch("The number of columns of X must match the length of y"))
    
    # TODO: Iterate more then once through the data?
    # TODO: Does it make a difference if a use the data in order or randomly?
    # TODO: Should a encode and hash the points beforehand, since I'll be using each
    #       multiple times?
    indices = collect(1:length(y)) # similar(y, Int)
    for _ in 1:epochs
        for i in shuffle!(indices) # eachindex(y)
            train!(d, view(X, :, i), y[i])
            # train!(d, X[:, i], y[i])
        end
    end

    nothing
end

# ================================ Functional ================================ #
# ---------------------------------------------------------------------------- #

struct FunctionalDiscriminator{L,T <: Real,E <: AbstractEncoder{T}}
    input_len::Int
    n::Int
    η::Float64
    # default::Tuple{Int,Float64}
    # scaling::Float64
    encoder::E
    segments::Vector{Int}
    offsets::Vector{Int}
    nodes::Vector{FunctionalNode}
end
    
function FunctionalDiscriminator(input_len::Int, n::Int, η::Float64, encoder::E, partitioner::Function; loss::Symbol=:mse_loss, seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    max_tuple_size = (UInt == UInt32) ? 32 : 64
    (n > max_tuple_size) && throw(DomainError(n, "Tuple size may not be greater then $max_tuple_size"))

    res = resolution(encoder)

    indices = partitioner(input_len, res, n; seed)
    segments, offsets = indices_to_segment_offset(indices, input_len, res)

    nodes = [FunctionalNode(; kargs...) for _ in 1:cld(input_len * res, n)]

    FunctionalDiscriminator{loss,T,E}(input_len, n, η, encoder, segments, offsets, nodes)
end
    
function FunctionalDiscriminator(input_len::Int, n::Int, encoder::E; η::Float64=0.1, loss::Symbol=:mse_loss, partitioner::Symbol=:uniform, seed::Union{Nothing,Int}=nothing, kargs...) where {T <: Real,E <: AbstractEncoder{T}}
    if partitioner == :uniform
    p_func = uniform_random_tuples
    elseif partitioner == :significance
        p_func = significance_aware_random_tuples
    else
        throw(DomainError(partitioner, "Unknown partitioning"))
    end

    FunctionalDiscriminator(input_len, n, η, encoder, p_func; loss, seed, kargs...)
end

# The functional gradient method amounts to the addition of a new weighted kernel to the model
# and possibly the adjustment of the weights of all previous kernels. The center
# of the new kernel is determined by the training sample. The weight is
# determined by the error term of the loss function and the training step size.
# The adjustment of the previous kernel weights is determined by the
# regularization term of the loss function

function kernel_weight(d::FunctionalDiscriminator{:mse_loss,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {T <: Real}
    d.η * (y - predict(d, x))
end

function add_kernel!(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, weight::Float64) where {L,T <: Real}
    length(x) != d.input_len && throw(DimensionMismatch("expected x's length to be $(d.input_len). Got $(length(x))"))
    
    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        train!(node, encode(d.encoder, x, segments, offsets), weight)
    end

    return nothing
end

# Functional gradient descent over quadratic error
function train!(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}, x::AbstractVector{T}, y::Float64) where {L,T <: Real}
    add_kernel!(d, x, kernel_weight(d, x, y))

    return nothing
end

function train!(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}, y::AbstractVector{Float64}; epochs=1) where {L,T <: Real}
    size(X, 1) != d.input_len && throw(DimensionMismatch("expected number of rows of X to be to be $(d.input_len). Got $(size(X, 1))"))
    size(X, 2) != length(y) && throw(DimensionMismatch("The number of columns of X must match the length of y"))

    indices = collect(1:length(y))
    for _ in 1:epochs
        for i in shuffle!(indices)
            train!(d, view(X, :, i), y[i])
        end
    end

    nothing
end

function predict(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}, x::AbstractVector{T}) where {L,T <: Real}
    cum_weight = zero(Float64)

    for (node, segments, offsets) in zip(d.nodes, Iterators.partition(d.segments, d.n), Iterators.partition(d.offsets, d.n))
        key = encode(d.encoder, x, segments, offsets)
        weight = predict(node, key)

        cum_weight += weight
    end

    return cum_weight / length(d.nodes)
end

function predict(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}, X::AbstractMatrix{T}) where {L,T <: Real}
    [predict(d, x) for x in eachcol(X)]
end

# Reset all nodes of the network (but keep the partitioning)
function reset!(d::FunctionalDiscriminator{L,T,<:AbstractEncoder{T}}) where {L,T <: Real}
    for node in d.nodes
        reset!(node)
    end

    return nothing
end