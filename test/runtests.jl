using ramnet
using Test

@testset "Random mapping" begin
    input_v = Bool[1, 0, 0, 1, 0]
    input_m = Bool[0 0 1 1 1; 1 0 0 0 0]

    partition_size = 2

    mapper = RandomMapper(length(input_v), partition_size)
    
    expected_len = ceil(Int, length(input_v) / partition_size)

    @test length(mapper) == expected_len
    @test length(map(mapper, input_v)) == expected_len
    @test length(map(mapper, input_m)) == expected_len

    @test eltype(map(mapper, input_v)) == typeof(input_v)
    @test eltype(map(mapper, input_m)) == typeof(input_m)

    @test length(collect(map(mapper, input_v))) == expected_len
    @test length(collect(map(mapper, input_m))) == expected_len
    @test length(collect(random_mapping(input_v, partition_size))) == expected_len
    @test length(collect(random_mapping(input_m, partition_size))) == expected_len
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
end
