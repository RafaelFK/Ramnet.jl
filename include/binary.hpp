#ifndef UTIL_BINARY_HPP
#define UTIL_BINARY_HPP

#include <cstddef>
#include <vector>
#include <numeric>

// Utility funcions to manipulate binary words
namespace util { namespace binary {

  // Decodes a binary word into a integer
  size_t decode(const std::vector<bool> input);
};};

#endif