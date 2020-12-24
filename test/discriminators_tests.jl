using ramnet.Utils: stack
using ramnet.Partitioners: RandomPartitioner, LinearPartitioner
using ramnet.Models.Nodes: RegressionNode, GeneralizedRegressionNode
using ramnet.Models: train!, predict
using ramnet.Models: Discriminator, BleachingDiscriminator

@testset "Classification Discriminator" begin
    all_active = ones(Bool, 9)
    one_off    = Bool[0, ones(Bool, 8)...]

    d = Discriminator(length(all_active), 3)

    train!(d, all_active)

    @test predict(d, all_active) == 3
    @test predict(d, one_off) == 2

    inputs = stack(all_active, one_off)
    @test predict(d, inputs) == [3, 2]

    # Discriminator with externally-instantiated partitioner
    partitioner   = RandomPartitioner(length(all_active), 3)
    d_partitioner = Discriminator(partitioner)
    train!(d_partitioner, all_active)
    @test predict(d_partitioner, all_active) == 3

    # One-shot instantiation and training
    duplicated_input = stack(all_active, all_active)
    one_shot_d_1 = Discriminator(all_active, partitioner)
    one_shot_d_2 = Discriminator(all_active, 3)
    one_shot_d_3 = Discriminator(duplicated_input, 3)

    @test predict(one_shot_d_1, all_active) == 3
    @test predict(one_shot_d_2, all_active) == 3
    @test predict(one_shot_d_3, all_active) == 3

    # Discriminators with bleaching
    b_d = BleachingDiscriminator(length(all_active), 3)

    train!(b_d, all_active)

    @test predict(b_d, all_active) == 3
    @test predict(b_d, one_off) == 2

    @test predict(b_d, all_active; b=1) == 0
    @test predict(b_d, one_off; b=1) == 0

    train!(b_d, one_off)

    @test predict(b_d, one_off) == 3
    @test predict(b_d, all_active; b=1) == 2
    @test predict(b_d, one_off; b=1) == 2
end

@testset "Original and Discounted Regression Discriminator" begin
    X_train = Bool[
        1 1 0 0 0 0
        1 1 1 0 0 0
        1 1 1 1 0 0
        1 1 1 1 1 0
        1 1 1 1 1 1
    ]

    y_train = Float64[1, 2, 3, 4, 5]

    X_test = Bool[
        0 0 0 0 0 0
        1 0 0 0 0 0
        1 1 0 0 0 0
        1 1 1 1 1 1
        1 1 1 0 0 0
    ]

    # As in the original regression discriminator
    y_test_original = Float64[7 / 4, 7 / 4, 22 / 9, 32 / 9, 23 / 9]

    partitioner = LinearPartitioner(6, 2)

    regressor = Discriminator{LinearPartitioner,RegressionNode}(partitioner; γ=1.0)

    train!(regressor, X_train, y_train)

    @test predict(regressor, X_test) == y_test_original

    # Discounted regression discriminator
    e(n; γ=0.7) = (1 - γ^n) / (1 - γ) # sum of weights
    function s(vs; γ=0.7) # Discounted sum
        su = 0.0
        for (i, v) in Iterators.enumerate(Iterators.reverse(vs))
            su += γ^(i - 1) * v
        end

        su
    end

    y_test_discounted = Float64[
        (s([1]) + s([1,2,3])) / (e(1) + e(3)),
        (s([1]) + s([1,2,3])) / (e(1) + e(3)),
        (s([1,2,3,4,5]) + s([1]) + s([1,2,3])) / (e(5) + e(1) + e(3)),
        (s([1,2,3,4,5]) + s([3,4,5]) + s([5])) / (e(5) + e(3) + e(1)),
        (s([1,2,3,4,5]) + s([2]) + s([1,2,3])) / (e(5) + e(1) + e(3))
    ]

    regressor = Discriminator{LinearPartitioner,RegressionNode}(partitioner; γ=0.7)

    train!(regressor, X_train, y_train)

    @test predict(regressor, X_test) ≈ y_test_discounted
end

@testset "Generalized Regression Discriminator" begin
    X_train = Bool[
        1 1 0 0 0 0
        1 1 1 0 0 0
        1 1 1 1 0 0
        1 1 1 1 1 0
        1 1 1 1 1 1
    ]

    y_train = Float64[1, 2, 3, 4, 5]

    X_test = Bool[
        0 0 0 0 0 0
        1 0 0 0 0 0
        1 1 0 0 0 0
        1 1 1 1 1 1
        1 1 1 0 0 0
    ]

    # Fixed α
    y_test = Float64[5.25 / 4, 5.25 / 4, 13.3125 / 6, 20.8125 / 6, 14.3125 / 6]

    partitioner = LinearPartitioner(6, 2)

    regressor = Discriminator{LinearPartitioner,GeneralizedRegressionNode}(partitioner; α=0.5)

    train!(regressor, X_train, y_train)

    @test predict(regressor, X_test) == y_test
end
