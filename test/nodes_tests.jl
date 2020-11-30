using ramnet.Models: train!, predict
using ramnet.Models.Nodes

using ramnet.Utils: stack

@testset "Nodes" begin
    # Standard Node
    all_active   = ones(Bool, 9)
    all_inactive = zeros(Bool, 9)

    std_n = DictNode{Vector{Bool}}()

    train!(std_n, all_active)

    @test predict(std_n, all_active) == 1
    @test predict(std_n, all_inactive) == 0
    @test predict(std_n, stack(all_active, all_inactive)) == [1, 0]

    # Accumulator Node
    acc_n = AccNode()

    train!(acc_n, all_active)

    @test predict(acc_n, all_active) == true
    @test predict(acc_n, all_active; b=1) == false
    
    train!(acc_n, all_active)

    @test predict(acc_n, all_active) == true
    @test predict(acc_n, all_active; b=1) == true
    
    train!(acc_n, all_inactive)

    @test predict(acc_n, stack(all_active, all_inactive)) == Bool[1, 1]
    @test predict(acc_n, stack(all_active, all_inactive); b=1) == Bool[1, 0]
end