using Ramnet.Partitioners: 
partition,
RandomPartitioner,
random_partitioning,
LinearPartitioner

@testset "Random partitioning (and generic methods)" begin
    input_v        = Bool[1, 0, 0, 1, 0]
    input_m        = Bool[0 0 1 1 1; 1 0 0 0 0]

    shorter_v      = Bool[1, 0, 1]
    longer_v       = Bool[1, 1, 1, 1, 1, 0, 0, 1]

    shorter_m      = Bool[1 0 1; 1 0 1]
    longer_m       = Bool[1 1 1 1 1 0 0 1; 1 1 1 1 1 0 0 1]

    partition_size = 2

    partitioner = RandomPartitioner(length(input_v), partition_size)
    
    expected_len = ceil(Int, length(input_v) / partition_size)


    @test eltype(partition(partitioner, input_v)) == typeof(input_v)
    @test eltype(partition(partitioner, input_m)) == typeof(input_m)

    @test length(partitioner) == expected_len
    @test length(partition(partitioner, input_v)) == expected_len
    @test length(partition(partitioner, input_m)) == expected_len

    @test length(collect(partition(partitioner, input_v))) == expected_len
    @test length(collect(partition(partitioner, input_m))) == expected_len
    @test length(collect(random_partitioning(input_v, partition_size))) == expected_len
    @test length(collect(random_partitioning(input_m, partition_size))) == expected_len

    @test_throws DimensionMismatch partition(partitioner, shorter_v)
    @test_throws DimensionMismatch partition(partitioner, longer_v)
    @test_throws DimensionMismatch partition(partitioner, shorter_m)
    @test_throws DimensionMismatch partition(partitioner, longer_m)
end

@testset "Linear partitioning" begin
    input_v = Int[1, 2, 3, 4, 5]
    input_m = Int[1 2 3 4 5; 6 7 8 9 10]

    partition_size = 3
    partitioner = LinearPartitioner(length(input_v), partition_size)

    @test collect(partition(partitioner, input_v)) == [
        [1, 2, 3],
        [4, 5]
    ]

    @test collect(partition(partitioner, input_m)) == [
        [1 2 3; 6 7 8],
        [4 5; 9 10]
    ]
end