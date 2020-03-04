#include "../include/dense_ram_node.hpp"

namespace ramnet {
  DenseRAMNode::DenseRAMNode(const size_t input_size) : input_size{input_size} {
    if (input_size > sizeof(size_t)*8)
      throw std::length_error {"Bit strings are limited to sizeof(size_t)*8 bits!"};
    memory.resize(1 << input_size);
  }

  // TODO: This could be inlined in the abstract class definition
  void DenseRAMNode::train(const std::vector<bool>& encoded_input) {
    if (encoded_input.size() > input_size)
      throw std::length_error {"Bit string must not exceed input_size"};
    train(util::binary::decode(encoded_input));
  }

  void DenseRAMNode::train(const size_t decoded_input) {
    memory[decoded_input] = true;
  }

  size_t DenseRAMNode::size() const { return memory.size(); }

  // TODO: This could be inlined in the abstract class definition
  size_t DenseRAMNode::hammingWeight() const {
    return std::accumulate(
      memory.begin(),
      memory.end(),
      (size_t) 0,
      [](auto x, auto y){ return x + y; }
    );
  }

  // TODO: This could be inlined in the abstract class definition
  bool DenseRAMNode::fire(const std::vector<bool>& encoded_input) const {
    if (encoded_input.size() > input_size)
      throw std::length_error {"Bit string must not exceed input_size"};
    return fire(util::binary::decode(encoded_input));
  }

  bool DenseRAMNode::fire(const size_t decoded_input) const {
    return memory[decoded_input];
  }
};