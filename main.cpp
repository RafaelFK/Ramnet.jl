#include <iostream>
#include "ram_node.hpp"
#include "ram_discriminator.hpp"

int neuron_test() {
  ramnet::RAMNode<3> neuron {};

  neuron.train(0b000);

  std::cout << "Trained pattern: " << neuron.fire(0b000) << std::endl;
  std::cout << "Untrained pattern: " << neuron.fire(0b001) << std::endl;
}

int main() {
  ramnet::RAMDiscriminator<8, 2> classifier;

  classifier.train(0b11100100);

  std::cout << "Score: " << classifier.score(0b11100100) << std::endl;
  std::cout << "Score: " << classifier.score(0b01111000) << std::endl;
}