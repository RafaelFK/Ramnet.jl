#ifndef RANDOM_MAPPER_HPP
#define RANDOM_MAPPER_HPP

#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>

#include <mapper.hpp>

namespace ramnet {
  class RandomMapper: public Mapper {
  public:
    RandomMapper(const size_t input_size, const size_t tuple_size);
    std::vector<size_t> getTuples(std::vector<bool> input) const override;

  private:
    const size_t input_size;
    const size_t tuple_size;
    const size_t n_tuples;

    std::vector<size_t> indexes;
  };
};
#endif