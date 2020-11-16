using ramnet
using Test

@testset "Random mapping" begin
    input_v        = Bool[1, 0, 0, 1, 0]
    input_m        = Bool[0 0 1 1 1; 1 0 0 0 0]

    shorter_v      = Bool[1, 0, 1]
    longer_v       = Bool[1, 1, 1, 1, 1, 0, 0, 1]

    shorter_m      = Bool[1 0 1; 1 0 1]
    longer_m       = Bool[1 1 1 1 1 0 0 1; 1 1 1 1 1 0 0 1]

    partition_size = 2

    mapper = RandomMapper(length(input_v), partition_size)
    
    expected_len = ceil(Int, length(input_v) / partition_size)


    @test eltype(map(mapper, input_v)) == typeof(input_v)
    @test eltype(map(mapper, input_m)) == typeof(input_m)

    @test length(mapper) == expected_len
    @test length(map(mapper, input_v)) == expected_len
    @test length(map(mapper, input_m)) == expected_len

    @test length(collect(map(mapper, input_v))) == expected_len
    @test length(collect(map(mapper, input_m))) == expected_len
    @test length(collect(random_mapping(input_v, partition_size))) == expected_len
    @test length(collect(random_mapping(input_m, partition_size))) == expected_len

    @test_throws DimensionMismatch map(mapper, shorter_v)
    @test_throws DimensionMismatch map(mapper, longer_v)
    @test_throws DimensionMismatch map(mapper, shorter_m)
    @test_throws DimensionMismatch map(mapper, longer_m)
end

@testset "Discriminator" begin
    all_active = ones(Bool, 9)
    one_off    = Bool[0, ones(Bool, 8)...]

    d = StandardDiscriminator(length(all_active), 3)

    train!(d, all_active)

    @test predict(d, all_active) == 3
    @test predict(d, one_off) == 2

    inputs = vcat(reshape(all_active, (1, :)), reshape(one_off, (1, :)))
    @test predict(d, inputs) == [3, 2]

    # Discriminator with externally-instantiated mapper
    mapper   = RandomMapper(length(all_active), 3)
    d_mapper = StandardDiscriminator(mapper)
    train!(d_mapper, all_active)
    @test predict(d_mapper, all_active) == 3

    # One-shot instantiation and training
    duplicated_input = reduce(vcat, map(v -> reshape(v, (1, :)), [all_active, all_active]))
    one_shot_d_1 = StandardDiscriminator(all_active, mapper)
    one_shot_d_2 = StandardDiscriminator(all_active, 3)
    one_shot_d_3 = StandardDiscriminator(duplicated_input, 3)

    @test predict(one_shot_d_1, all_active) == 3
    @test predict(one_shot_d_2, all_active) == 3
    @test predict(one_shot_d_3, all_active) == 3
end

@testset "MultiDiscriminatorClassifier" begin
    all_active   = ones(Bool, 9)
    all_inactive = zeros(Bool, 9)
    
    # TODO: Make a Utils module with a stack function?
    X = reduce(vcat, map(v -> reshape(v, (1, :)), [all_active, all_inactive]))
    y = String["On", "Off"]

    model = MultiDiscriminatorClassifier{String}(9, 3)

    train!(model, all_active, "On")
    train!(model, all_inactive, "Off")

    @test predict(model, all_active) == "On"
    @test predict(model, all_inactive) == "Off"

    # train!(model, X, y)

    # @test predict(model, all_active) == "On"
    # @test predict(model, all_inactive) == "Off"

    # Future tests:
    # - predict over model trained on a single pattern
    # - predict over a model that hasn't been trained at all
end
