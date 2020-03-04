#ifndef RAM_DISCRIMINATOR_HPP
#define RAM_DISCRIMINATOR_HPP

#include <cstddef>
#include <vector>
#include <algorithm>

#include "dense_ram_node.hpp"

namespace ramnet {

  class RAMDiscriminator {

  public:
    // TODO: Either enforce that input_size must be a multiple of tuple_size or
    //       work around it
    RAMDiscriminator(const size_t input_size, const size_t tuple_size);

    void train(const std::vector<bool>& input);
    void train(const size_t decoded_input);

    size_t score(const std::vector<bool>& input);
    size_t score(const size_t decoded_input);

  private:
    const size_t input_size;
    const size_t tuple_size;
    const size_t layer_size;

    std::vector<DenseRAMNode> layer;

    // Here, a linear mapping is assumed. By that, a mean that the first
    // tuple_size bits are mapped to the first neuron, the second tuple_size bits
    // are mapped to the second neuron and so on
    // TODO: This is most likely gonna break if input_size is not a multiple of
    //       tuple_size
    std::vector<std::vector<bool>> getTuples(
      const std::vector<bool>& input
    );
  };
};

#endif