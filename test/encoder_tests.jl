using ramnet.Encoders: Thermometer, CircularThermometer, encode!, encode

@testset "Simple Thermometer" begin
    ## Thermometer instantiation

    @test_throws ArgumentError Thermometer(0.0, -20.0, 10)
    @test_throws DomainError Thermometer(0.0, 1.0, -2)

    ## Encoding scalars

    thermo = Thermometer(0.0, 1.0, 10)
    out_pattern = Vector{Bool}(undef, 10)

    encode!(thermo, 0.5, out_pattern)
    
    @test out_pattern == Bool[1,1,1,1,1,0,0,0,0,0]
    @test_throws DimensionMismatch encode!(thermo, 0.5, Vector{Bool}(undef, 5))
    
    @test encode(thermo, 0.5) == Bool[1,1,1,1,1,0,0,0,0,0]
    @test encode(thermo, -1.1) == zeros(Bool, 10)
    @test encode(thermo, 2.0) == ones(Bool, 10)

    ## Encoding vectors

    out_pattern = Matrix{Bool}(undef, 10, 2)
    encode!(thermo, [0.0, 0.5], out_pattern)

    @test out_pattern == hcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )
    @test_throws DimensionMismatch encode!(thermo, [0.0, 0.5], Matrix{Bool}(undef, 5, 2))
    @test_throws DimensionMismatch encode!(thermo, [0.0, 0.5], Matrix{Bool}(undef, 10, 1))

    @test encode(thermo, [0.0, 0.5]; flat=false) == hcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )
    @test encode(thermo, [-1.1, 0.5]; flat=false) == hcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )
    @test encode(thermo, [2.0, 0.5]; flat=false) == hcat(
      ones(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )

    ### Flat outputs
    out_pattern = Vector{Bool}(undef, 20)
    encode!(thermo, [0.0, 0.5], out_pattern)

    @test out_pattern == vcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )

    @test_throws DimensionMismatch encode!(thermo, [0.0, 0.5], Vector{Bool}(undef, 10))

    @test encode(thermo, [0.0, 0.5]) == vcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )
    @test encode(thermo, [-1.1, 0.5]) == vcat(
      zeros(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )
    @test encode(thermo, [2.0, 0.5]) == vcat(
      ones(Bool, 10),
      Bool[1,1,1,1,1,0,0,0,0,0]
    )

    ## Encoding matrices (one observation per row)

    out_pattern = Array{Bool,3}(undef, 10, 2, 2)
    encode!(thermo, [0.0 0.5; 0.7 1.0], out_pattern)

    @test out_pattern == cat(
        hcat(
          zeros(Bool, 10),
          Bool[1,1,1,1,1,0,0,0,0,0]
        ),
        hcat(
          Bool[1,1,1,1,1,1,1,0,0,0],
          ones(Bool, 10)
        );
        dims=3
    )
    @test_throws DimensionMismatch encode!(thermo, [0.0 0.5; 0.7 1.0], Array{Bool,3}(undef, 5, 2, 2))
    @test_throws DimensionMismatch encode!(thermo, [0.0 0.5; 0.7 1.0], Array{Bool,3}(undef, 10, 1, 2))
    @test_throws DimensionMismatch encode!(thermo, [0.0 0.5; 0.7 1.0], Array{Bool,3}(undef, 10, 2, 1))

    @test encode(thermo, [-1.0 0.5; 0.7 1.0]; flat=false) == cat(
        hcat(
          zeros(Bool, 10),
          Bool[1,1,1,1,1,0,0,0,0,0]
        ),
        hcat(
          Bool[1,1,1,1,1,1,1,0,0,0],
          ones(Bool, 10)
        );
        dims=3
    )

    ### Flat outputs

    out_pattern = Matrix{Bool}(undef, 2, 20)
    encode!(thermo, [0.0 0.5; 0.7 1.0], out_pattern)

    @test out_pattern ==  vcat(
        reshape(vcat(
          zeros(Bool, 10),
          Bool[1,1,1,1,1,0,0,0,0,0]
        ), (1, :)),
        reshape(vcat(
          Bool[1,1,1,1,1,1,1,0,0,0],
          ones(Bool, 10)
        ), (1, :))
    )
    @test_throws DimensionMismatch encode!(thermo, [0.0 0.5; 0.7 1.0], Matrix{Bool}(undef, 1, 20))
    @test_throws DimensionMismatch encode!(thermo, [0.0 0.5; 0.7 1.0], Matrix{Bool}(undef, 2, 10))

    @test encode(thermo, [-1.0 0.5; 0.7 1.0]) == vcat(
        reshape(vcat(
          zeros(Bool, 10),
          Bool[1,1,1,1,1,0,0,0,0,0]
        ), (1, :)),
        reshape(vcat(
          Bool[1,1,1,1,1,1,1,0,0,0],
          ones(Bool, 10)
        ), (1, :))
    )

end

@testset "Circular Thermometer" begin
    @test_throws ArgumentError CircularThermometer(10, 1, 5)
    @test_throws DomainError CircularThermometer(0, 1, 1)

    thermo = CircularThermometer(Float64, 0, 1, 5)

    @test encode(thermo, 0.0) == Bool[1, 1, 0, 0, 0]
    @test encode(thermo, 0.2) == Bool[0, 1, 1, 0, 0]
    @test encode(thermo, 0.4) == Bool[0, 0, 1, 1, 0]
    @test encode(thermo, [0.2, 0.4]) == Bool[0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    @test encode(thermo, [0.2, 0.4];flat=false) == Bool[
      0 0
      1 0
      1 1
      0 1
      0 0
    ]

    @test encode(thermo, [
      0.0 0.2
      0.4 0.6
    ]) == Bool[
      1 1 0 0 0 0 1 1 0 0
      0 0 1 1 0 0 0 0 1 1
    ]

    @test encode(thermo, [
      0.0 0.2
      0.4 0.6
    ]; flat=false) == cat(
      [
        1 0
        1 1
        0 1
        0 0
        0 0
      ],
      [
        0 0
        0 0
        1 0
        1 1
        0 1
      ];
      dims=3
    )
end