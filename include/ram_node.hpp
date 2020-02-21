#ifndef RAM_NODE_HPP
#define RAM_NODE_HPP

#include <cstddef>
#include <bitset>
#include <vector>
#include <numeric>

#include "binary.hpp"

namespace ramnet {
  class RAMNode {
  public:
    RAMNode(const size_t input_size);
    
    void train(const std::vector<bool>& encoded_input);
    void train(const size_t decoded_input);

    size_t size() const;
    size_t hammingWeight() const;

    bool fire(const std::vector<bool>& encoded_input) const;
    bool fire(const size_t decoded_input) const;

  private:
    const size_t input_size;
    std::vector<bool> memory;
  };
};

#endif