#include "../include/ram_discriminator.hpp"

namespace ramnet {
  // TODO: Either enforce that input_size must be a multiple of tuple_size or
  //       work around it
  RAMDiscriminator::RAMDiscriminator(const size_t input_size, const size_t tuple_size)
  : input_size{input_size}, tuple_size{tuple_size}, layer_size{input_size/tuple_size} {
    layer.reserve(layer_size);
    for(size_t i = 0; i != layer_size; ++i)
      layer.emplace_back(tuple_size);
  }

  void RAMDiscriminator::train(const std::vector<bool>& input) {
    // Map input into tuples to be feed to the neurons
    auto tuples = getTuples(input);

    for (size_t i = 0; i != layer_size; ++i)
      layer[i].train(tuples[i]);
  }

    void RAMDiscriminator::train(const size_t decoded_input) {
      train(util::binary::encode(decoded_input, input_size));
    }

    size_t RAMDiscriminator::score(const std::vector<bool>& input) {
      size_t s = 0;
      auto tuples = getTuples(input);

      for (size_t i = 0; i != layer_size; ++i)
        s += (size_t)layer[i].fire(tuples[i]);

      return s;
    }

    size_t RAMDiscriminator::score(const size_t decoded_input) {
      return score(util::binary::encode(decoded_input, input_size));
    }

    // Here, a linear mapping is assumed. By that, a mean that the first
    // tuple_size bits are mapped to the first neuron, the second tuple_size bits
    // are mapped to the second neuron and so on
    // TODO: This is most likely gonna break if input_size is not a multiple of
    //       tuple_size
    std::vector<std::vector<bool>> RAMDiscriminator::getTuples(
      const std::vector<bool>& input
    ) {
      std::vector<std::vector<bool>> tuples{layer_size};
 
      for(size_t i = 0; i != layer_size; ++i)
        std::copy(
          input.rbegin() + i*tuple_size,
          input.rbegin() + (i+1)*tuple_size,
          std::back_inserter(tuples[layer_size - i - 1])
        );
      
      return tuples;
    }
}