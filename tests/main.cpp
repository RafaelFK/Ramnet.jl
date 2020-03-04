#include <iostream>
#include "../include/ram_node.hpp"
#include "../include/ram_discriminator.hpp"

int main() {
  ramnet::DenseRAMNode neuron {3};

  std::cout << neuron.size() << std::endl;
  std::cout << "Hamming Weight: " << neuron.hammingWeight() << std::endl;
  
  neuron.train(0b000);

  std::cout << "Trained pattern: " << neuron.fire(0b000) << std::endl;
  std::cout << "Untrained pattern: " << neuron.fire(0b001) << std::endl;
}

int discriminator_test() {
  ramnet::RAMDiscriminator classifier {8, 2};

  classifier.train(0b11100100);

  std::cout << "Score: " << classifier.score(0b11100100) << std::endl;
  std::cout << "Score: " << classifier.score(0b01111000) << std::endl;
}