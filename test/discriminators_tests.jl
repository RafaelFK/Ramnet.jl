using ramnet.Utils: stack

@testset "Discriminator" begin
    all_active = ones(Bool, 9)
    one_off    = Bool[0, ones(Bool, 8)...]

    d = StandardDiscriminator(length(all_active), 3)

    train!(d, all_active)

    @test predict(d, all_active) == 3
    @test predict(d, one_off) == 2

    inputs = stack(all_active, one_off)
    @test predict(d, inputs) == [3, 2]

    # Discriminator with externally-instantiated mapper
    mapper   = RandomMapper(length(all_active), 3)
    d_mapper = StandardDiscriminator(mapper)
    train!(d_mapper, all_active)
    @test predict(d_mapper, all_active) == 3

    # One-shot instantiation and training
    duplicated_input = stack(all_active, all_active)
    one_shot_d_1 = StandardDiscriminator(all_active, mapper)
    one_shot_d_2 = StandardDiscriminator(all_active, 3)
    one_shot_d_3 = StandardDiscriminator(duplicated_input, 3)

    @test predict(one_shot_d_1, all_active) == 3
    @test predict(one_shot_d_2, all_active) == 3
    @test predict(one_shot_d_3, all_active) == 3
end
