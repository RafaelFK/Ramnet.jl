# TODO: Consider other concept for the names here instead of mapping (ex. partitioning)
module Mappers

abstract type Mapper end

include("RandomMapper.jl")
export RandomMapper, random_mapping

end