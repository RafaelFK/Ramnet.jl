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