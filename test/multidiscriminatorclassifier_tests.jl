@testset "MultiDiscriminatorClassifier" begin
    all_active   = ones(Bool, 9)
    all_inactive = zeros(Bool, 9)
    
    # TODO: Make a Utils module with a stack function?
    X = mapreduce(permutedims, vcat, [all_active, all_inactive])
    y = String["On", "Off"]

    # Training with single inputs
    model_1 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_1, all_active, "On")
    train!(model_1, all_inactive, "Off")

    @test predict(model_1, all_active) == "On"
    @test predict(model_1, all_inactive) == "Off"

    # Training with multiple inputs
    model_2 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_2, X, y)

    @test predict(model_2, X) == ["On", "Off"]
    
    # train!(model, X, y)

    # @test predict(model, all_active) == "On"
    # @test predict(model, all_inactive) == "Off"

    # Future tests:
    # - predict over model trained on a single pattern
    # - predict over a model that hasn't been trained at all
end