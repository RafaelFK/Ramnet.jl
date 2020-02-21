#include "../include/binary.hpp"

namespace util { namespace binary {
  size_t decode(const std::vector<bool> input){
    return std::accumulate(
      input.rbegin(),
      input.rend(),
      0,
      [](auto x, auto y) { return (x << 1) + y; }
    );
  }
};};