#ifndef RAM_DISCRIMINATOR_HPP
#define RAM_DISCRIMINATOR_HPP

#include <cstddef>
#include <vector>
#include <algorithm>

#include "ram_node.hpp"

namespace ramnet {

  class RAMDiscriminator {

  public:
    // TODO: Either enforce that input_size must be a multiple of tuple_size or
    //       work around it
    RAMDiscriminator(const size_t input_size, const size_t tuple_size)
    : input_size{input_size}, tuple_size{tuple_size}, layer_size{input_size/tuple_size} {
      layer.reserve(layer_size);
      for(size_t i = 0; i != layer_size; ++i)
        layer.emplace_back(tuple_size);
    }
    void train(const std::vector<bool>& input) {
      // Map input into tuples to be feed to the neurons
      auto tuples = getTuples(input);

      for (size_t i = 0; i != layer_size; ++i)
        layer[i].train(tuples[i]);
    }

    void train(const size_t decoded_input) {
      auto input = std::vector<bool> {};
      
    }

    size_t score(const std::vector<bool>& input) {
      size_t s = 0;
      auto tuples = getTuples(input);

      for (size_t i = 0; i != layer_size; ++i)
        s += (size_t)layer[i].fire(tuples[i]);

      return s;
    }

  private:
    const size_t input_size;
    const size_t tuple_size;
    const size_t layer_size;

    std::vector<RAMNode> layer;

    // Here, a linear mapping is assumed. By that, a mean that the first
    // tuple_size bits are mapped to the first neuron, the second tuple_size bits
    // are mapped to the second neuron and so on
    // TODO: This is most likely gonna break if input_size is not a multiple of
    //       tuple_size
    std::vector<std::vector<bool>> getTuples(
      const std::vector<bool>& input
    ) {
      std::vector<std::vector<bool>> tuples{layer_size};
 
      for(size_t i = 0; i != layer_size; ++i)
        std::copy(
          input.rbegin() + i*tuple_size,
          input.rbegin() + (i+1)*tuple_size,
          std::back_inserter(tuples[layer - i - 1])
        );
      
      return tuples;
    }
  };
};

#endif