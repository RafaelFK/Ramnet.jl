#include <random_mapper.hpp>

namespace ramnet {
  RandomMapper::RandomMapper(const size_t input_size, const size_t tuple_size)
  : input_size{input_size},
    tuple_size{tuple_size},
    n_tuples{input_size / tuple_size},
    indexes(input_size) {
    if ((input_size % tuple_size) != 0)
      throw std::logic_error {"input_size must be a multiple of tuple_size"};

    // Building the random mapping
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), std::mt19937{std::random_device{}()});
  }

  std::vector<size_t> RandomMapper::getTuples(std::vector<bool> input) const {
    std::vector<size_t> result(n_tuples);

    for (size_t i = 0; i != n_tuples; ++i) {
      for (size_t j = 0; j != tuple_size; ++j) {
        result[i] += (input[indexes[tuple_size*i+j]] << 1*j); 
      }
    }

    return result;
  }
};