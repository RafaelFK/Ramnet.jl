#include "../include/ram_node.hpp"

namespace ramnet {
  template<typename RAM_T>
  RAMNode<RAM_T>::RAMNode(const size_t input_size) : input_size{input_size} {
    memory.resize(1 << input_size);
  }

  template<typename RAM_T>
  void RAMNode<RAM_T>::train(const std::vector<bool>& encoded_input) {
    train(util::binary::decode(encoded_input));
  }

  template<typename RAM_T>
  void RAMNode<RAM_T>::train(const size_t decoded_input) {
    memory[decoded_input] = true;
  }

  template<typename RAM_T>
  size_t RAMNode<RAM_T>::size() const { return memory.size(); }


  template<typename RAM_T>
  size_t RAMNode<RAM_T>::hammingWeight() const {
    return std::accumulate(
      memory.begin(),
      memory.end(),
      (size_t) 0,
      [](auto x, auto y){ return x + y; }
    );
  }

  template<typename RAM_T>
  bool RAMNode<RAM_T>::fire(const std::vector<bool>& encoded_input) const {
    return fire(util::binary::decode(encoded_input));
  }

  template<typename RAM_T>
  bool RAMNode<RAM_T>::fire(const size_t decoded_input) const {
    return memory[decoded_input];
  }

  // Allowed specializations
  template class RAMNode<std::vector<bool>>;
};