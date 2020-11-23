using ramnet.Utils: stack, ambiguity

@testset "Utilities" begin
    # Stacking vectors
    @test stack([1,2,3]) == [1 2 3]
    @test stack([1,2,3], [4,5,6]) == [1 2 3; 4 5 6]
    @test stack([1 2 3], [4 5 6]) == [1 2 3; 4 5 6]
    # @test stack([1 2 3], [4,5,6]) == [1 2 3; 4 5 6]

    @test_throws DimensionMismatch stack([1], [2, 3, 4])
    @test_throws DimensionMismatch stack([1 2], [3 4 5])

    # Measuring discriminator response ambiguity
    @test ambiguity([3,3,0,2,2]) == 0.0
    @test ambiguity([3,3,0,2,2], [0,0,0,2,2]) == 0.6
end