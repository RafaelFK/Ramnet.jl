using ramnet.Utils: stack
using ramnet.Models: train!, predict, predict_response, predict_bleached_response, predict_bleached
using ramnet.Models: MultiDiscriminatorClassifier, BleachingDiscriminator

@testset "MultiDiscriminatorClassifier" begin
    all_active   = ones(Bool, 9)
    all_inactive = zeros(Bool, 9)

    X = stack(all_active, all_inactive)
    y = String["On", "Off"]

    ## Training with single inputs
    model_1 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_1, all_active, "On")
    train!(model_1, all_inactive, "Off")

    @test predict(model_1, all_active) == "On"
    @test predict_response(model_1, all_active) == Dict("On" => 3, "Off" => 0)
    @test predict(model_1, all_inactive) == "Off"
    @test predict_response(model_1, all_inactive) == Dict("On" => 0, "Off" => 3)

    ## Training with multiple inputs
    model_2 = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model_2, X, y)

    @test predict(model_2, X) == ["On", "Off"]
    @test predict_response(model_2, X) == Dict("On" => [3,0], "Off" => [0,3])

    ## Dimensions must match when training with multiple inputs
    model_3 = MultiDiscriminatorClassifier{String}(9, 3)

    @test_throws DimensionMismatch train!(model_3, X, ["On"])

    ## Classifiers with nodes that support bleaching
    model_b = MultiDiscriminatorClassifier{String,BleachingDiscriminator}(9, 3)

    train!(model_b, X, y)

    @test predict(model_b, X) == ["On", "Off"]
    @test predict_response(model_b, X) == Dict("On" => [3,0], "Off" => [0,3])
    @test predict_response(model_b, X; b=1) == Dict("On" => [0, 0], "Off" => [0, 0])

    train!(model_b, all_active, "On")

    @test predict_response(model_b, X; b=1) ==  Dict("On" => [3, 0], "Off" => [0, 0])

    ## Linear search bleaching
    model_b = MultiDiscriminatorClassifier{String,BleachingDiscriminator}(9, 3)

    train!(model_b, X, y)
    # Simulating ambiguity. "Off" target is also trained with the all_active input
    train!(model_b, stack(all_active, all_active), ["On", "Off"])

    @test predict_bleached_response(model_b, all_active) == Dict("On" => 3, "Off" => 0)
    @test predict_bleached_response(model_b, all_inactive) == Dict("On" => 0, "Off" => 3)
    @test predict_bleached(model_b, X) == ["On", "Off"]


    # Future tests:
    # - predict over model trained on a single pattern
    # - predict over a model that hasn't been trained at all
end