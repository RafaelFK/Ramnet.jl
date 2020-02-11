#ifndef RAM_NODE_HPP
#define RAM_NODE_HPP

#include <cstddef>
#include <bitset>

namespace ramnet {
  template<size_t input_size>
  class RAMNode {
  public:
    void train(const std::bitset<input_size>& input) {
      memory.set(input.to_ullong());
    }

    bool fire(const std::bitset<input_size>& input) const {
      return memory[input.to_ullong()];
    }
  private:
    std::bitset<(1 << input_size)> memory {};
  };
};

#endif