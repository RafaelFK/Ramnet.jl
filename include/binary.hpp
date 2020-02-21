#ifndef UTIL_BINARY_HPP
#define UTIL_BINARY_HPP

#include <cstddef>
#include <vector>
#include <numeric>

// Utility funcions to manipulate binary words
namespace util { namespace binary {

  // Decodes a binary word into a integer
  size_t decode(const std::vector<bool> input);

  // Encode a integer integer into a binary word
  std::vector<bool> encode(const size_t input, const size_t length);
};};

#endif