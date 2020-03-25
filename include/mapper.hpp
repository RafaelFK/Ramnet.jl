#ifndef MAPPER_HPP
#define MAPPER_HPP

#include <cstddef>
#include <vector>

namespace ramnet {
  class Mapper {
  public:
    virtual std::vector<size_t> getTuples(std::vector<bool> input) const = 0;
  };
};

#endif