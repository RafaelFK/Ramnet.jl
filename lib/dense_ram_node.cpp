#include "../include/dense_ram_node.hpp"

namespace ramnet {
  DenseRAMNode::DenseRAMNode(const size_t input_size) : input_size{input_size} {
    memory.resize(1 << input_size);
  }

  // TODO: This could be inlined in the abstract class definition
  void DenseRAMNode::train(const std::vector<bool>& encoded_input) {
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
    return fire(util::binary::decode(encoded_input));
  }

  bool DenseRAMNode::fire(const size_t decoded_input) const {
    return memory[decoded_input];
  }
};