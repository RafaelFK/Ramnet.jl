#ifndef NODE_HPP
#define NODE_HPP

#include <cstddef>
#include <vector>

namespace ramnet {
  class Node {
  public:
    virtual void train(const std::vector<bool>& encoded_input) = 0;
    virtual void train(const size_t decoded_input) = 0;

    virtual size_t size() const = 0;
    virtual size_t hammingWeight() const = 0;

    virtual bool fire(const std::vector<bool>& encoded_input) const = 0;
    virtual bool fire(const size_t decoded_input) const = 0;
  };
};

#endif