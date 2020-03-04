#include "../include/sparse_ram_node.hpp"

namespace ramnet {
  SparseRAMNode::SparseRAMNode(const size_t input_size):
    input_size{input_size} {};

  // TODO: This could be inlined in the abstract class definition
  void SparseRAMNode::train(const std::vector<bool>& encoded_input) {
    train(util::binary::decode(encoded_input));
  }

  void SparseRAMNode::train(const size_t decoded_input) {
    memory[decoded_input] = true;
  }

  size_t SparseRAMNode::size() const {
    return ((size_t)1 << input_size);
  }

  size_t SparseRAMNode::hammingWeight() const {
    return memory.size();
  }

  // TODO: This could be inlined in the abstract class definition
  bool SparseRAMNode::fire(const std::vector<bool>& encoded_input) const {
    return fire(util::binary::decode(encoded_input));
  }

  bool SparseRAMNode::fire(const size_t decoded_input) const {
    return memory.count(decoded_input);
  }
};