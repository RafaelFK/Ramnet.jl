#include "../catch.hpp"
#include <binary.hpp>

#include <vector>

using namespace util::binary;

TEST_CASE ("decode is little endian", "[decode]") {
  REQUIRE(decode(std::vector<bool>{0,0,0,1,1,0,1,1}) == 216);
  REQUIRE(decode(std::vector<bool>{1,1,1,0,0,1,0,0}) == 39);
  REQUIRE(decode(std::vector<bool>{1,0,1,0,1,0,1,1,1,0,0,1}) == 2517);
}