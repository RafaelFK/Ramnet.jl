#include "ram_node.hpp"

namespace ramnet {
  RAMNode::RAMNode(const size_t input_size) : input_size{input_size} {
    memory.resize(1 << input_size);
  }

  void RAMNode::train(const std::vector<bool>& encoded_input) {
    train(util::binary::decode(encoded_input));
  }

  void RAMNode::train(const size_t decoded_input) {
    memory[decoded_input] = true;
  }

  size_t RAMNode::size() const { return memory.size(); }

  size_t RAMNode::hammingWeight() const {
    return std::accumulate(
      memory.begin(),
      memory.end(),
      (size_t) 0,
      [](auto x, auto y){ return x + y; }
    );
  }

  bool RAMNode::fire(const std::vector<bool>& encoded_input) const {
    return fire(util::binary::decode(encoded_input));
  }
  bool RAMNode::fire(const size_t decoded_input) const {
    return memory[decoded_input];
  }
};