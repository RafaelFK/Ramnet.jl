#ifndef RAM_DISCRIMINATOR_HPP
#define RAM_DISCRIMINATOR_HPP

#include <cstddef>
#include <bitset>
#include <array>

#include "ram_node.hpp"

namespace ramnet {

  template<size_t input_size, size_t tuple_size>
  class RAMDiscriminator {
    static_assert(
      input_size % tuple_size == 0, 
      "input_size must be a multiple of tuple_size"
    );

  public:

    void train(const std::bitset<input_size>& input) {
      // Map input into tuples to be feed to the neurons
      auto tuples = getTuples(input);

      for (size_t i = 0; i != input_size / tuple_size; ++i)
        layer[i].train(tuples[i]);
    }

    size_t score(const std::bitset<input_size>& input) {
      size_t s = 0;
      auto tuples = getTuples(input);

      for (size_t i = 0; i != input_size / tuple_size; ++i)
        s += (size_t)layer[i].fire(tuples[i]);

      return s;
    }

  private:
    std::array<RAMNode<tuple_size>, input_size / tuple_size> layer;

    // Here, a linear mapping is assumed. By that, a mean that the first
    // tuple_size bits are mapped to the first neuron, the second tuple_size bits
    // are mapped to the second neuron and so on
    std::array<std::bitset<tuple_size>, input_size / tuple_size> getTuples(
      const std::bitset<input_size>& input
    ) {
      std::array<std::bitset<tuple_size>, input_size / tuple_size> tuples;
 
      // Iterate over the array backwards
      for (size_t i = 0; i != input_size / tuple_size; ++i)
        tuples[input_size / tuple_size - i - 1] = (input >> i*tuple_size).to_ullong();

      return tuples;
    }
  };
};

#endif