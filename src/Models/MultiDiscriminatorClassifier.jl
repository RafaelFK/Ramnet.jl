# TODO: Allow the instantiation without the specification of width (Could be
#       determined from the training data)
# TODO: Check how other classification models handle the target type. Should I 
#       simply enforce it to be an integer?
struct MultiDiscriminatorClassifier{C}
    mapper::RandomMapper
    discriminators::Dict{C,StandardDiscriminator}

    function MultiDiscriminatorClassifier{C}(width::Int, n::Int; seed::Union{Nothing,Int}=nothing) where C
        mapper = RandomMapper(width, n; seed)

        new{C}(mapper, Dict{C,StandardDiscriminator}())
    end
end

# TODO: Use the get! method with the default argument
function train!(model::MultiDiscriminatorClassifier{C}, X::T, y::C) where {T <: AbstractVector{Bool},C}       
    # if haskey(model.discriminators, y)
    #     train!(model.discriminators[y], X)
    # else
    #     # TODO: Allow the instantiation and training of a discriminator in a single shot
    #     model.discriminators[y] = StandardDiscriminator(model.mapper)
    #     train!(model.discriminators[y], X)
    # end

    # TODO: Allow the instantiation and training of a discriminator in a single shot
    train!(get!(model.discriminators, y) do
        StandardDiscriminator(model.mapper)
    end, X)
end

# TODO: Check if model was trained
function predict(model::MultiDiscriminatorClassifier{C}, X::T) where {T <: AbstractVector{Bool},C}
    largest_response = -1
    best_category = first(keys(model.discriminators))

    for (category, discriminator) in model.discriminators
        response = predict(discriminator, X)

        if response > largest_response
            largest_response = response
            best_category = category
        end
    end

    return best_category
end