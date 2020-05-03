#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// #include <iostream>
// #include <exception>

// #include "../include/dense_ram_node.hpp"
// #include "../include/sparse_ram_node.hpp"
// #include "../include/ram_discriminator.hpp"
// #include "../include/random_mapper.hpp"
// #include "../include/binary.hpp"

// int main() {
//   try {
//     ramnet::RandomMapper m {8, 2};

//     // Trying out a pattern
//     auto tuples = m.getTuples(std::vector<bool>{0,0,0,1,1,0,1,1});

//     for (auto tuple : tuples) {
//       std::cout << "[*] " << tuple << std::endl; 
//     }

//   } catch (std::exception& e) {
//     std::cout << e.what() << std::endl;
//   }
// }

// int neuron_test() {
//   std::cout << sizeof(size_t) << std::endl;

//   try {
//     ramnet::DenseRAMNode neuron {8};

//     std::cout << neuron.size() << std::endl;
//     std::cout << "Hamming Weight: " << neuron.hammingWeight() << std::endl;

//     neuron.train(0b000);

//     std::cout << "Trained pattern: " << neuron.fire(0b000) << std::endl;
//     std::cout << "Untrained pattern: " << neuron.fire(0b001) << std::endl;

//   } catch (std::exception& e) {
//     std::cout << e.what() << std::endl;
//   }
// }

// int discriminator_test() {
//   ramnet::RAMDiscriminator classifier {8, 2};

//   classifier.train(0b11100100);

//   std::cout << "Score: " << classifier.score(0b11100100) << std::endl;
//   std::cout << "Score: " << classifier.score(0b01111000) << std::endl;
// }