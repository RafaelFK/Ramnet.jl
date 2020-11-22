using ramnet.Utils: stack

@testset "Utilities" begin
    @test stack([1,2,3]) == [1 2 3]
    @test stack([1,2,3], [4,5,6]) == [1 2 3; 4 5 6]
    @test stack([1 2 3], [4 5 6]) == [1 2 3; 4 5 6]
    # @test stack([1 2 3], [4,5,6]) == [1 2 3; 4 5 6]

    @test_throws DimensionMismatch stack([1], [2, 3, 4])
    @test_throws DimensionMismatch stack([1 2], [3 4 5])
end