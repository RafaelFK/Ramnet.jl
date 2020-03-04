#include "../include/binary.hpp"

namespace util { namespace binary {
  size_t decode(const std::vector<bool> input) {
    return std::accumulate(
      input.rbegin(),
      input.rend(),
      0,
      [](auto x, auto y) { return (x << 1) + y; }
    );
  }

  std::vector<bool> encode(size_t input, const size_t length) {
    auto bits = std::vector<bool> {};
    bits.reserve(length);
    
    for(size_t i = 0; i != length; ++i)
      bits.push_back((input >> i) %2);

    return bits;
  }
};};