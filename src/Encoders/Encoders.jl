module Encoders

export AbstractEncoder, encode

abstract type AbstractEncoder end

function encode! end
function encode end

########################### Generic encoding methods ###########################
"""
    encode(encoder::E, x::T) where {E <: AbstractEncoder,T <: Real}

Encode `x` as a binary vector according to `encoder`.

# Examples
```jldoctest
julia> encode(Thermometer(Float64, 0, 1, 5), 0.5)
5-element Array{Bool,1}:
 1
 1
 1
 0
 0

 julia> encode(CircularThermometer(Float64, 0, 1, 5), 0.5)
 5-element Array{Bool,1}:
 0
 0
 1
 1
 0
```
"""
function encode(encoder::E, x::T) where {E <: AbstractEncoder,T <: Real}
    pattern = Vector{Bool}(undef, encoder.resolution)

    encode!(encoder, x, pattern)
end

"""
    encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}

Encode `x` as a binary matrix, where each component of `x` is mapped into a
column, according to `encoder` and place the result in `pattern`.

# Examples
```jldoctest
julia> encode!(Thermometer(Float64, 0, 1, 5), [0.1, 0.5], Matrix{Bool}(undef, 5, 2))
5×2 Array{Bool,2}:
 1  1
 0  1
 0  1
 0  0
 0  0

julia> encode!(CircularThermometer(Float64, 0, 1, 5), [0.1, 0.5], Matrix{Bool}(undef, 5, 2))
5×2 Array{Bool,2}:
 1  0
 1  0
 0  1
 0  1
 0  0
```
"""
function encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, x, pattern)

    if encoder.min isa AbstractVector
        for (i, col) in Iterators.enumerate(eachcol(pattern))
            encode!(encoder, encoder.min[i], encoder.max[i], x[i], col)
        end
    else
        for (i, col) in Iterators.enumerate(eachcol(pattern))
            encode!(encoder, x[i], col)
        end
    end

    pattern
end

"""
    encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}

Encode `x` as a binary vector, where each component of `x` is mapped into a
partition of the binary vector, according to `encoder` and place the result in
`pattern`.

# Examples
```jldoctest
julia> encode!(Thermometer(Float64, 0, 1, 5), [0.1, 0.5], Vector{Bool}(undef, 10))
10-element Array{Bool,1}:
 1
 0
 0
 0
 0
 1
 1
 1
 0
 0

julia> encode!(CircularThermometer(Float64, 0, 1, 5), [0.1, 0.5], Vector{Bool}(undef, 10))
10-element Array{Bool,1}:
 1
 1
 0
 0
 0
 0
 0
 1
 1
 0
```
"""
function encode!(encoder::E, x::AbstractVector{T}, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, x, pattern)

    if encoder.min isa AbstractVector
        for (i, slice) in Iterators.enumerate(Iterators.partition(pattern, encoder.resolution))
            encode!(encoder, encoder.min[i], encoder.max[i], x[i], slice)
        end
    else
        for (i, slice) in Iterators.enumerate(Iterators.partition(pattern, encoder.resolution))
            encode!(encoder, x[i], slice)
        end
    end

    pattern
end

"""
    encode(encoder::E, x::AbstractVector{T}; flat=true) where {E <: AbstractEncoder,T <: Real}

Encode `x` according to `encoder`.

If keyword argument `flat` is `true`, the result is as a binary vector, where
each component of `x` is mapped into a partition of the output vector. If `flat`
is false, the result is a binary matrix, where each component is mapped to a
column of the output matrix.

# Examples
```jldoctest
julia> encode(Thermometer(Float64, 0, 1, 5), [0.1, 0.5]; flat=true)
10-element Array{Bool,1}:
 1
 0
 0
 0
 0
 1
 1
 1
 0
 0

 julia> encode(Thermometer(Float64, 0, 1, 5), [0.1, 0.5]; flat=false)
 5×2 Array{Bool,2}:
  1  1
  0  1
  0  1
  0  0
  0  0
```
"""
function encode(encoder::E, x::AbstractVector{T}; flat=true) where {E <: AbstractEncoder,T <: Real}
    if flat
        pattern = Vector{Bool}(undef, encoder.resolution * length(x))
    else
        pattern = Matrix{Bool}(undef, encoder.resolution, length(x))
    end

    encode!(encoder, x, pattern)
end

"""
    encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {E <: AbstractEncoder,T <: Real}

Encode each row of `X` according to `encoder`.

Each row of `X` is encoded as a binary matrix and mapped to a slice of `pattern`.

# Examples
```jldoctest
julia> encode!(Thermometer(Float64, 0, 1, 5), [0 0.1; 0.3 0.7], Array{Bool,3}(undef, 5, 2, 2))
5×2×2 Array{Bool,3}:
[:, :, 1] =
 0  1
 0  0
 0  0
 0  0
 0  0

[:, :, 2] =
 1  1
 1  1
 0  1
 0  1
 0  0

```
"""
function encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, X, pattern)

    for (slice, row) in Iterators.zip(eachslice(pattern; dims=3), eachrow(X))
        encode!(encoder, row, slice)
    end

    pattern
end

# Flat version. All values in each row of the input matrix are encoded and placed in a single row of the output matrix
"""
    encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}

Encode `X` according to `encoder`.

Each row of `X` is encoded as a binary vector and mapped to a row of `pattern`.

# Examples
```jldoctest
julia> encode!(Thermometer(Float64, 0, 1, 5), [0 0.1; 0.3 0.7], Matrix{Bool}(undef, 2, 10))
2×10 Array{Bool,2}:
 0  0  0  0  0  1  0  0  0  0
 1  1  0  0  0  1  1  1  1  0

```
"""
function encode!(encoder::E, X::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    validate_input_output(encoder, X, pattern)

    for (slice, row) in Iterators.zip(eachrow(pattern), eachrow(X))
        encode!(encoder, row, slice)
    end

    pattern
end

"""
    encode(encoder::E, X::AbstractMatrix{T}; flat=true) where {E <: AbstractEncoder,T <: Real}

Encode `x` according to `encoder`.

If keyword argument `flat` is `true`, the result is as a binary matrix, where
each component of row `X` is mapped into a row of the output matrix. If `flat`
is false, the result is a 3-dimensional binary array, where each row is mapped
to a slice of the output array.

# Examples
```jldoctest
julia> encode(Thermometer(Float64, 0, 1, 5), [0 0.1; 0.3 0.7]; flat=true)
2×10 Array{Bool,2}:
 0  0  0  0  0  1  0  0  0  0
 1  1  0  0  0  1  1  1  1  0

julia> encode(Thermometer(Float64, 0, 1, 5), [0 0.1; 0.3 0.7]; flat=false)
5×2×2 Array{Bool,3}:
[:, :, 1] =
 0  1
 0  0
 0  0
 0  0
 0  0

[:, :, 2] =
 1  1
 1  1
 0  1
 0  1
 0  0
```
"""
function encode(encoder::E, X::AbstractMatrix{T}; flat=true) where {E <: AbstractEncoder,T <: Real}
    if flat
        pattern = Matrix{Bool}(undef, size(X, 1), encoder.resolution * size(X, 2))
    else
        pattern = Array{Bool,3}(undef, encoder.resolution, size(X, 2), size(X, 1))
    end

    encode!(encoder, X, pattern)
end

########################## Generic validation methods ##########################
function validate_input_output(encoder::E, x::T, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    # If more then one minimum/maximum were especified, it doesn't make sense to try to encode a scalar
    length(encoder.min) != 1 && throw(DomainError(x, "expected vector of length $(length(encoder.min))"))

    length(pattern) != encoder.resolution && throw(
      DimensionMismatch(
        "output pattern must have the length of the thermometer's resolution"))

    nothing
end

function validate_input_output(encoder::E, x::AbstractVector{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
      DimensionMismatch(
        "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != length(x) && throw(
      DimensionMismatch(
          "the number of columns of the output pattern must equal x's length"))

    encoder.min isa AbstractVector && length(x) != length(encoder.min) && throw(
      DimensionMismatch(
        "the length of x must match the lengths of the min and max vectors"))

    nothing
end

function validate_input_output(encoder::E, x::AbstractVector{T}, pattern::AbstractVector{Bool}) where {E <: AbstractEncoder,T <: Real}
    length(pattern) != length(x) * encoder.resolution && throw(
      DimensionMismatch(
        "the length of the output pattern must equal the thermometer's resolution times the length of x"))

    encoder.min isa AbstractVector && length(x) != length(encoder.min) && throw(
      DimensionMismatch(
        "the length of x must match the lengths of the min and max vectors"))
    nothing
end

function validate_input_output(encoder::E, X::AbstractMatrix{T}, pattern::AbstractArray{Bool,3}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != encoder.resolution && throw(
      DimensionMismatch(
        "the number of rows of the output pattern must equal the thermometer's resolution"))

    size(pattern, 2) != size(X, 2) && throw(
      DimensionMismatch(
          "the number of columns of the output pattern must equal X's number of columns"))

    size(pattern, 3) != size(X, 1) && throw(
      DimensionMismatch(
          "the depth of the output pattern must equal X's number of rows"))

    encoder.min isa AbstractVector && size(X, 2) != length(encoder.min) && throw(
      DimensionMismatch(
        "the number of columns of X must match the lengths of the min and max vectors"))

    nothing
end

function validate_input_output(encoder::E, X::AbstractMatrix{T}, pattern::AbstractMatrix{Bool}) where {E <: AbstractEncoder,T <: Real}
    size(pattern, 1) != size(X, 1) && throw(
        DimensionMismatch(
          "the number of rows of the output pattern must equal the the number of rows of X"))
    
    size(pattern, 2) != size(X, 2) * encoder.resolution && throw(
          DimensionMismatch(
              "the number of columns of the output pattern must equal the thermometer's resolution times X's number of columns"))

    encoder.min isa AbstractVector && size(X, 2) != length(encoder.min) && throw(
      DimensionMismatch(
        "the number of columns of X must match the lengths of the min and max vectors"))

    nothing
end

include("Thermometer.jl")
export Thermometer

include("CircularThermometer.jl")
export CircularThermometer

end