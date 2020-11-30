using ramnet.Utils: stack
using ramnet.Models: train!, predict, predict_response
using ramnet.Models: MultiDiscriminatorClassifier

@testset "MultiDiscriminatorClassifier" begin
    all_active   = ones(Bool, 9)
    all_inactive = zeros(Bool, 9)

    X = stack(all_active, all_inactive)
    y = String["On", "Off"]

    # Training with single inputs
    model_1 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_1, all_active, "On")
    train!(model_1, all_inactive, "Off")

    @test predict(model_1, all_active) == "On"
    @test predict_response(model_1, all_active) == Dict("On" => 3, "Off" => 0)
    @test predict(model_1, all_inactive) == "Off"
    @test predict_response(model_1, all_inactive) == Dict("On" => 0, "Off" => 3)

    # Training with multiple inputs
    model_2 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_2, X, y)

    @test predict(model_2, X) == ["On", "Off"]
    @test predict_response(model_2, X) == Dict("On" => [3,0], "Off" => [0,3])

    # Dimensions must match when training with multiple inputs
    model_3 = MultiDiscriminatorClassifier{String}(9, 3)

    @test_throws DimensionMismatch train!(model_3, X, ["On"])

    # Future tests:
    # - predict over model trained on a single pattern
    # - predict over a model that hasn't been trained at all
end